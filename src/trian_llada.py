import torch
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk, DatasetDict
import argparse

# Pretraining code snippet
def forward_process(input_ids, eps=1e-3):
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    # 126336 is used for [MASK] token
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    return noisy_batch, masked_indices, p_mask

def custom_pretrain_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    Custom loss function for SFTTrainer that implements the masked language modeling approach.
    
    Args:
        model: The model being trained
        inputs: Dictionary containing input tensors
        return_outputs: Whether to return model outputs along with the loss
    
    Returns:
        Loss tensor or tuple of (loss, outputs) if return_outputs is True
    """
    input_ids = inputs["input_ids"]
    
    # Forward process to add noise
    noisy_batch, masked_indices, p_mask = forward_process(input_ids)
    
    # Get model outputs
    outputs = model(input_ids=noisy_batch)
    logits = outputs.logits
    
    token_loss = F.cross_entropy(
        logits[masked_indices],
        input_ids[masked_indices], 
        eduction='none'
    ) / p_mask[masked_indices]
    ce_loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
    
    return (ce_loss, outputs) if return_outputs else ce_loss



def custom_sft_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
# def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None)
    """
    Custom loss function for SFTTrainer that implements the masked language modeling approach.
    
    Args:
        model: The model being trained
        inputs: Dictionary containing input tensors
        return_outputs: Whether to return model outputs along with the loss
    
    Returns:
        Loss tensor or tuple of (loss, outputs) if return_outputs is True
    """
    input_ids = inputs["input_ids"]
    prompt_lengths = inputs["prompt_lengths"]
    
    # Forward process to add noise
    noisy_batch, masked_indices, p_mask = forward_process(input_ids)
    
    # Do not add noise to the prompt
    token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
    prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    
    # Calculate the answer length (including the padded <EOS> tokens)
    prompt_mask = prompt_mask.to(torch.int64)    
    answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
    answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
    
    masked_indices = (noisy_batch == 126336)
    
    # Get model outputs
    outputs = model(input_ids=noisy_batch)
    logits = outputs.logits
    
    # Calculate loss
    token_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1))[masked_indices.view(-1)], 
        input_ids.view(-1)[masked_indices.view(-1)], 
        reduction='none'
    ) / p_mask.view(-1)[masked_indices.view(-1)]
    
    ce_loss = torch.sum(token_loss / answer_lengths.view(-1)[masked_indices.view(-1)]) / input_ids.shape[0]
    
    return (ce_loss, outputs) if return_outputs else ce_loss

def preprocess_function(examples, tokenizer, max_length=4096):
    """Process the dataset for training"""
    # This is a simplified example - adjust based on your actual data structure
    prompts = examples["prompt"]
    responses = examples["response"]
    
    # Create conversation format that can be used with chat template
    conversations = []
    for prompt, response in zip(prompts, responses):
        conversations.append([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ])
    
    # Tokenize prompts to get their lengths (for masking purposes)
    prompt_tokens = tokenizer(prompts, truncation=True, max_length=max_length)
    prompt_lengths = torch.tensor([len(tokens) for tokens in prompt_tokens["input_ids"]])
    
    # Apply chat template and tokenize
    formatted_texts = []
    for conversation in conversations:
        formatted_text = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        formatted_texts.append(formatted_text)
    
    tokenized = tokenizer(formatted_texts, truncation=True, max_length=max_length, padding="longest")
    
    result = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "prompt_lengths": prompt_lengths,  # Add prompt lengths to the result
    }
    return result

def main():
    parser = argparse.ArgumentParser(description="Train a language model with masked language modeling")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="Model name or path")
    parser.add_argument("--dataset_name", type=str, default="nvidia/HelpSteer2", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--loss_type", type=str, choices=['sft', 'pretrain'], default='sft', 
                      help="Type of loss function to use: 'sft' or 'pretrain'")
    args = parser.parse_args()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Make sure the tokenizer has a padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset(args.dataset_name)
    
    # Preprocess the dataset
    train_dataset = dataset["train"].map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False
    )

    # Initialize the SFTTrainer with your custom loss function
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Override the compute_loss method with your chosen loss function
    trainer.compute_loss = custom_sft_loss if args.loss_type == 'sft' else custom_pretrain_loss

    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")

if __name__ == "__main__":
    main()