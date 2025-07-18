import argparse
import torch
import os
from datetime import timedelta
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from tqdm import tqdm
from transformers import set_seed
from typing import List
# from transformers import AutoModelForCausalLM
from transformers import LlamaForCausalLM
import transformers
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import math
from accelerate.utils import (
    InitProcessGroupKwargs,
    set_seed,
    DummyOptim,
    DummyScheduler,
)
from pathlib import Path
from typing import Optional, Tuple
import glob
import random
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

from packed_dataset import CombinedDataset, PackedDataset
from easy_context import (
    prepare_seq_parallel_inputs,
    apply_seq_parallel_monkey_patch,
    prepare_dataloader,
    apply_unsloth_offloaded_gradient_checkpoint_monkey_patch,
)

apply_unsloth_offloaded_gradient_checkpoint_monkey_patch()

# Option 1: Popular text datasets
# train_data_config = [
#    ("openwebtext", 0.6),  # OpenWebText dataset
#    ("bookcorpus", 0.4),   # BookCorpus dataset
# ]

# Option 2: Instruction/chat datasets
train_data_config = [
    ("DKYoon/SlimPajama-6B", 1.0),  # SlipPajama dataset
    # ("HuggingFaceH4/ultrachat_200k", 0.7),  # UltraChat dataset
    # ("microsoft/orca-math-word-problems-200k", 0.3),  # Orca Math
]

# Option 3: Code datasets
# train_data_config = [
#    ("bigcode/the-stack", 0.8),  # The Stack code dataset
#    ("codeparrot/github-code", 0.2),  # GitHub code
# ]

val_data_config = None


def transition(x_0, sigma, maskable_mask, mask_token_id):
    # move_chance = 1 - (-sigma).exp()
    move_chance = sigma
    print(f"move_chance: {move_chance}")
    move_indices = (
        torch.rand(*x_0.shape, device=x_0.device) < move_chance
    ) & maskable_mask
    print(f"move_indices: {move_indices}")
    x_t = torch.where(move_indices, mask_token_id, x_0)
    return x_t


def create_dataloader(
    batch_size: int,
    block_size: int,
    dataset: str,
    dataset_weights: List[float],
    accelerator,
    tokenizer,
    shuffle: bool = True,
    seed: int = 4756,
    split="train",
    max_tokens: int = None,
) -> DataLoader:
    datasets = []
    #data_config = train_data_config if "train" in split else val_data_config
    #print(data_config)
    for dataset_name in dataset:
        # Load dataset from Hugging Face Hub
        print(dataset_name)
        hf_dataset = load_dataset(
            dataset_name, split=split, streaming=True, trust_remote_code=True
        )

        # Convert HF dataset to PackedDataset format
        dataset = PackedDataset.from_hf_dataset(
            hf_dataset=hf_dataset,
            tokenizer=tokenizer,
            #n_chunks=8,
            n_chunks=1,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed + accelerator.process_index,
            num_processes=accelerator.num_processes,
            process_rank=accelerator.process_index,
            max_tokens=max_tokens,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No datasets found in config. Check your train_data_config."
        )

    sum_weights = sum(dataset_weights)
    weights = [el / sum_weights for el in dataset_weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )


def create_dataloaders(
    batch_size: int,
    block_size: int,
    accelerator,
    tokenizer,
    train_dataset: str = "data/redpajama_sample",
    train_split: str = "train",
    val_dataset: Optional[str] = None,
    val_split: Optional[str] = None,
    seed: int = 12345,
    max_tokens: int = None,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        accelerator=accelerator,
        dataset=[train_dataset],
        dataset_weights=[1.0],
        tokenizer=tokenizer,
        shuffle=True,
        seed=seed,
        split=train_split,
        max_tokens=max_tokens,
        # split="train_sft"
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            accelerator=accelerator,
            dataset=[val_dataset],
            dataset_weights=[1.0],
            tokenizer=tokenizer,
            shuffle=False,
            seed=seed,
            split=val_split,
            max_tokens=max_tokens,
        )
        if val_dataset
        else None
    )
    return train_dataloader, val_dataloader


def main(args):
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.wandb_key:
        import wandb

        wandb.login(key=args.wandb_key)
    set_seed(args.seed)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))

    # Initialize accelerator - let it handle distributed setup automatically
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        log_with="wandb" if args.wandb_key else None,
        kwargs_handlers=[timeout],
        # fsdp_plugin=fsdp_plugin,
    )

    # Remove manual distributed initialization - let Accelerate/DeepSpeed handle it

    # accelerator.init_trackers(project_name=args.wandb_key, init_kwargs={"wandb":{"name":args.output_dir.split("/")[-1]}})
    accelerator.init_trackers(
        project_name="AR-to-Diffusion",
        init_kwargs={"wandb": {"name": args.output_dir.split("/")[-1]}},
    )
    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    # Create tokenizer from the model
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Add padding token if it doesn't exist (needed for LLaMA models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_dataloader = create_dataloaders(
        batch_size=args.batch_size,
        block_size=args.seq_length,
        accelerator=accelerator,
        tokenizer=tokenizer,
        train_dataset=args.dataset,
        val_dataset=args.dataset,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=3407,
        max_tokens=args.max_tokens,
    )

    model = LlamaForCausalLM.from_pretrained(
        args.model,
        # device_map=accelerator.device,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    )

    model_type = (
        "llama" if isinstance(model, transformers.LlamaForCausalLM) else "mistral"
    )
    apply_seq_parallel_monkey_patch(args.parallel_mode, model_type)

    if args.learning_rate != 2e-5:
        accelerator.print(
            f"Warning: You also need to modify accelerate_configs/zero3_offload.json to change the learning rate"
        )

    # Check if we're using DeepSpeed
    use_deepspeed = (
        hasattr(accelerator.state, "deepspeed_plugin")
        and accelerator.state.deepspeed_plugin is not None
    )

    if use_deepspeed:
        # Use DummyOptim/DummyScheduler for DeepSpeed
        optim = DummyOptim(model.parameters(), lr=args.learning_rate)
        scheduler = DummyScheduler(
            optim,
            num_training_steps=args.max_train_steps,
            total_num_steps=args.max_train_steps,
        )
    else:
        # Use regular optimizers for non-DeepSpeed training
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        optim = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=0,
            num_training_steps=args.max_train_steps,
        )

    model, optim, scheduler = accelerator.prepare(model, optim, scheduler)
    train_loader = prepare_dataloader(args.parallel_mode, train_loader, accelerator)
    model.gradient_checkpointing_enable()

    # accelerator.register_for_checkpointing(scheduler)

    accelerator.print(f"Max train steps: {args.max_train_steps}")
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    model.train()
    loss_func = CrossEntropyLoss(inplace_backward=True, reduction="none")

    sampling_eps = 1e-3
    mask_token_id = (
        args.mask_token
    )  # mask token id. can be a new token or an existing token.

    for step, batch in enumerate(train_loader):
        print(f"-------step: {step}---------")
        input_ids = batch[..., : args.seq_length + 1]
        # print(input_ids.shape)
        target_ids = batch[..., : args.seq_length + 1]
        position_ids = (
            torch.arange(args.seq_length + 1)
            .unsqueeze(0)
            .expand(input_ids.shape[0], -1)
        )
        # shard the input_ids according to the world size and rank according to zig zag attention

        prepared = prepare_seq_parallel_inputs(
            args.parallel_mode,
            input_ids,
            position_ids,
            target_ids,
            accelerator.process_index,
            accelerator.num_processes,
            accelerator.device,
        )
        local_input_ids = prepared["local_input_ids"]
        local_position_ids = prepared["local_position_ids"]
        local_target_ids = prepared["local_target_ids"]
        src_mask = torch.zeros_like(
            local_input_ids, dtype=torch.bool, device=local_input_ids.device
        )

        t = (1 - sampling_eps) * torch.rand(
            local_input_ids.shape[0], device=local_input_ids.device
        ) + sampling_eps
        #t = torch.clamp(t, max=0.25)  # Cap t elements to be less than 0.5
        #t = torch.clamp(t, max=0.1)  # Cap t elements to be less than 0.5
        t = torch.clamp(t, max=0.01)  # Cap t elements to be less than 0.5
        print(f"t: {t}")
        sigma = t
        dsigma = torch.reciprocal(t)  # dsigma = 1 / t

        local_input_ids = transition(
            local_input_ids,
            sigma[:, None],
            maskable_mask=~src_mask,
            mask_token_id=mask_token_id,
        )
        loss_log = None
        loss_mask = local_input_ids == mask_token_id
        with accelerator.accumulate(model):
            logits = model(
                local_input_ids,
                position_ids=local_position_ids,
            ).logits

            logits = logits[:, :-1]
            loss_mask = loss_mask[:, 1:]
            local_target_ids = local_target_ids[:, 1:]
            loss = loss_func(
                logits.reshape(-1, logits.shape[-1]), local_target_ids.reshape(-1)
            ).reshape(local_target_ids.shape[0], -1)
            loss = loss.masked_fill(~loss_mask, 0)
            loss = (dsigma[:, None] * loss).sum() / loss_mask.sum()  # avg token loss
            print(f"loss: {loss}")
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                # pay attention here. When any seq parallel algo is turned on. This technically only log the very first chunk's loss
                # and what is the first chunk really depends on how do you shard the sequence
                # for zig zag attention, the first chunk contains the left most and rightmost tokens
                # so you cannot compare the (logged) loss of dist attention and zigzag ring attention.
                # loss_log = {"loss": loss.item(), "ppl": math.exp(loss.item())}

                # we now try gathered loss to verify if ring attention and dist flash attention produce the same loss
                # this may slow down the training
                gathered_loss = accelerator.reduce(loss.clone().detach(), "mean")
                try:
                    ppl = math.exp(gathered_loss.item())
                except OverflowError:
                    ppl = float("inf")  # or use a large number like 1e10
                loss_log = {
                    "loss": gathered_loss.item(),
                    "ppl": ppl,
                }
                accelerator.log(loss_log, step=completed_steps)

            optim.step()
            scheduler.step()
            optim.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            if loss_log is not None:
                progress_bar.set_postfix(loss_log)
            completed_steps += 1

        if completed_steps >= args.max_train_steps:
            break

    accelerator.print(f"Training Finished")
    accelerator.end_training()

    if args.output_dir is not None:
        accelerator.print(f"Saving model to {args.output_dir}")

        # Only wait for everyone if we're in a distributed setting
        if accelerator.num_processes > 1:
            accelerator.wait_for_everyone()

        # For single GPU, use simpler saving approach to avoid DeepSpeed issues
        if accelerator.num_processes == 1:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                state_dict=accelerator.get_state_dict(model),
            )
            tokenizer.save_pretrained(args.output_dir)
        else:
            state_dict = accelerator.get_state_dict(model)
            accelerator.unwrap_model(model).save_pretrained(
                f"{args.output_dir}",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=state_dict,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)

        accelerator.print(f"Saving Finished")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--wandb-key", type=str)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int, default=400)
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    args.add_argument(
        "--dataset",
        type=str,
        default="/work/nvme/bbzy/shivama2/TinyLlama/data/slim_star_combined/",
    )  # Path to processed dataset from TinyLlama pre-processing.
    args.add_argument("--train-split", type=str, default="train")
    args.add_argument("--val-dataset", type=str, default=None)
    args.add_argument("--val-split", type=str, default="validation")
    args.add_argument("--seq-length", type=int, default=16384)
    args.add_argument("--mask-token", type=int, default=811)
    args.add_argument("--max-tokens", type=int, default=None)
    args.add_argument(
        "--parallel_mode",
        type=str,
        choices=["dist_flash_attn", "ulysses_attn", "data_parallel"],
    )
    main(args.parse_args())
