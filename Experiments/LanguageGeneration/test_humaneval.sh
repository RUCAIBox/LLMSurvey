cd HumanEval
python generate.py --model gpt-3.5-turbo --path ChatGPT
evalplus.evaluate --dataset humaneval --samples humaneval/gpt-3.5-turbo_temp_0.1
python generate.py --model text-davinci-003 --path ChatGPT
evalplus.evaluate --dataset humaneval --samples humaneval/text-davinci-003_temp_0.1
python generate.py --model text-davinci-002 --path ChatGPT
evalplus.evaluate --dataset humaneval --samples humaneval/text-davinci-002_temp_0.1
python generate.py --model claude --path Claude
evalplus.evaluate --dataset humaneval --samples humaneval/claude_temp_0.1
python generate.py --model vicuna-7b --path {vicuna7b_model_path}
evalplus.evaluate --dataset humaneval --samples humaneval/vicuna-7b_temp_0.1
python generate.py --model alpaca-7b --path {alpaca7b_model_path}
evalplus.evaluate --dataset humaneval --samples humaneval/alpaca-7b_temp_0.1
python generate.py --model chatglm --path {chatglm6b_model_path}
evalplus.evaluate --dataset humaneval --samples humaneval/chatglm_temp_0.1
python generate.py --model llama-7b --path {llama7b_model_path}
evalplus.evaluate --dataset humaneval --samples humaneval/llama-7b_temp_0.1
python generate.py --model falcon-7b --path {falcon7b_model_path}
evalplus.evaluate --dataset humaneval --samples humaneval/falcon-7b_temp_0.1
python generate.py --model pythia-12b --path {pythia12b_model_path}
evalplus.evaluate --dataset humaneval --samples humaneval/pythia-12b_temp_0.1
python generate.py --model pythia-6.9b --path {pythia6.9b_model_path}
evalplus.evaluate --dataset humaneval --samples humaneval/pythia-6.9b_temp_0.1
cd ..
