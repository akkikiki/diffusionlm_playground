#python src/infer.py --prompt "Write a Python script." --base_model_name  meta-llama/Llama-3.2-3B-Instruct --model_name meta-llama/Llama-3.2-3B-Instruct
#python src/infer.py --prompt "What is " --base_model_name meta-llama/Llama-3.2-3B-Instruct --model_name meta-llama/Llama-3.2-3B-Instruct
#python src/infer.py --prompt "What is " --base_model_name meta-llama/Llama-3.1-8B-Instruct --model_name meta-llama/Llama-3.1-8B-Instruct --verbose True
#python src/infer.py --prompt "What is 1 + 1?" --base_model_name meta-llama/Llama-3.1-8B-Instruct --model_name meta-llama/Llama-3.1-8B-Instruct --verbose True
python src/infer.py --prompt "What is 1 + 1?" --base_model_name meta-llama/Llama-3.1-8B-Instruct --model_name meta-llama/Llama-3.1-8B-Instruct
#python src/infer.py --unconditional --base_model_name meta-llama/Llama-3.2-3B-Instruct --model_name meta-llama/Llama-3.2-3B-Instruct
