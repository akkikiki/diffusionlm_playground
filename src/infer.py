import torch
from transformers import AutoConfig, AutoTokenizer
from argparse import ArgumentParser


from model import DiscreteDiffusionModel, generate_samples

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default='diffusionfamily/diffugpt-s')
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

    # Some patching for AR models to work with masked filling out of the box
    config.task_specific_params = {}
    config.task_specific_params["text-generation"] = {
      "do_sample": True,
      #"max_length": 512
      #"max_length": 8
      #"max_length": 16
      #"max_length": 32
      "max_length": 64 
    }
    #tokenizer.mask_token = "ти"
    #tokenizer.mask_token_id = tokenizer.encode("ти")[1]
    #tokenizer.mask_token = "<|reserved_special_token_247|>"
    MASK_TOKEN = "<|eot_id|>"
    tokenizer.mask_token = MASK_TOKEN
    tokenizer.mask_token_id = tokenizer.encode(MASK_TOKEN)[1]

    
    # model = DiscreteDiffusionModel(args.base_model_name, config, tokenizer)
    model = DiscreteDiffusionModel.from_pretrained(
        pretrained_model_name_or_path=model_name, 
        config=config, 
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
        
        # Use apply_chat_template to format the prompt
        messages = [{"role": "user", "content": args.prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
        )
        print(f"formatted_prompt: {formatted_prompt}")
        prefix = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        
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
    # if not args.unconditional and args.prompt:
    #     print("\n" + "="*20, "Example with conditional generation...")
    #     
    #     # Also use apply_chat_template for the example
    #     example_messages = [{"role": "user", "content": "Today is a wonderful day,"}]
    #     example_formatted = tokenizer.apply_chat_template(
    #         example_messages, 
    #         tokenize=False, 
    #         add_generation_prompt=True
    #     )
    #     prefix = tokenizer.encode(example_formatted)

    #     src_mask = [1]*len(prefix)+[0]*(gen_len-len(prefix))
    #     x0 = prefix + [0]*(gen_len-len(prefix))

    #     inputs = {
    #         "input_ids": torch.tensor([x0]), 
    #         "src_mask": torch.tensor([src_mask])
    #     }
    #     res = generate_samples(model, args, tokenizer, inputs, verbose=args.verbose)
    #     pred = tokenizer.decode(res.tolist()[0])
    #     print(pred)
