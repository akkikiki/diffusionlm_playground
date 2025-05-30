import torch
from transformers import AutoConfig, AutoTokenizer
import torch.nn.functional as F
from argparse import ArgumentParser


from model import DiscreteDiffusionModel, generate_samples

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default='diffusionfamily/diffugpt-s')
    parser.add_argument("--base_model_name", type=str, default='gpt2')
    parser.add_argument("--shift", type=bool, default=True) # do not change this
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--logits_temp", type=float, default=0.95)
    parser.add_argument("--topp_temp", type=float, default=0.9)
    parser.add_argument("--verbose", type=bool, default=False) # print middle state
    parser.add_argument("--prompt", type=str, default="", help="Text prompt for conditional generation")
    parser.add_argument("--unconditional", action="store_true", help="Generate unconditionally (ignore prompt)")

    args = parser.parse_args()

    # model_name = 'gpt2'  # 'gpt2-medium', 'gpt2-large'
    model_name = args.model_name
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # model = DiscreteDiffusionModel(args.base_model_name, config, tokenizer)
    model = DiscreteDiffusionModel.from_pretrained(
        model_name, 
        model=args.base_model_name, 
        config=config, 
        tokenizer=tokenizer,
        device='cuda'
    ).to('cuda')

    # import pdb; pdb.set_trace();

    gen_len = config.task_specific_params['text-generation']['max_length']
    print("="*20, "Generating...", gen_len)
    
    if args.unconditional or not args.prompt:
        # un-conditional generation
        print("="*20, "Unconditional gen...")
        x0 = [0] * gen_len
        inputs = {
            "input_ids": torch.tensor([x0])
        }
        res = generate_samples(model, args, tokenizer, inputs, verbose=args.verbose)
        pred = tokenizer.decode(res.tolist()[0])
        print(pred)
    else:
        # conditional generation with custom prompt
        print("="*20, f"Conditional gen with prompt: '{args.prompt}'")
        prefix = [tokenizer.bos_token_id] + tokenizer.encode(args.prompt)
        
        # Ensure prefix doesn't exceed generation length
        if len(prefix) >= gen_len:
            print(f"Warning: Prompt is too long ({len(prefix)} tokens), truncating to {gen_len-1} tokens")
            prefix = prefix[:gen_len-1]

        src_mask = [1]*len(prefix)+[0]*(gen_len-len(prefix))
        x0 = prefix + [0]*(gen_len-len(prefix))

        inputs = {
            "input_ids": torch.tensor([x0]), 
            "src_mask": torch.tensor([src_mask])
        }
        res = generate_samples(model, args, tokenizer, inputs, verbose=args.verbose)
        pred = tokenizer.decode(res.tolist()[0])
        print(pred)

    # Optional: Keep the original example for demonstration
    if not args.unconditional and args.prompt:
        print("\n" + "="*20, "Example with default prompt...")
        prefix = [tokenizer.bos_token_id] + tokenizer.encode("Today is a wonderful day,")

        src_mask = [1]*len(prefix)+[0]*(gen_len-len(prefix))
        x0 = prefix + [0]*(gen_len-len(prefix))

        inputs = {
            "input_ids": torch.tensor([x0]), 
            "src_mask": torch.tensor([src_mask])
        }
        res = generate_samples(model, args, tokenizer, inputs, verbose=args.verbose)
        pred = tokenizer.decode(res.tolist()[0])
        print(pred)
