
# ChatGPT
python ChatGPT.py --api_key your_api_key --input data/openbook_valid.jsonl --output output/openbook_valid_predict_chatgpt.jsonl
python ChatGPT.py --api_key your_api_key --input data/hellaswag_valid.jsonl --output output/hellaswag_valid_predict_chatgpt.jsonl
python ChatGPT.py --api_key your_api_key --input data/socialiqa_valid.jsonl --output output/socialiqa_valid_predict_chatgpt.jsonl

# Claude
python Claude.py --channel_id xxxxxxx --userOAuthToken xxxxxxx --input data/openbook_valid.jsonl --output output/openbook_valid_predict_claude.jsonl
python Claude.py --channel_id xxxxxxx --userOAuthToken xxxxxxx --input data/hellaswag_valid.jsonl --output output/hellaswag_valid_predict_claude.jsonl
python Claude.py --channel_id xxxxxxx --userOAuthToken xxxxxxx --input data/socialiqa_valid.jsonl --output output/socialiqa_valid_predict_claude.jsonl

# Davinci003
python davinci-003.py --api_key xxxxxxx --input data/openbook_valid.jsonl --output output/openbook_valid_predict_davinci_003.jsonl
python davinci-003.py --api_key xxxxxxx --input data/hellaswag_valid.jsonl --output output/hellaswag_valid_predict_davinci_003.jsonl
python davinci-003.py --api_key xxxxxxx --input data/socialiqa_valid.jsonl --output output/socialiqa_valid_predict_davinci_003.jsonl

# Davinci002
python davinci-002.py --api_key xxxxxxx --input data/openbook_valid.jsonl --output output/openbook_valid_predict_davinci_002.jsonl
python davinci-002.py --api_key xxxxxxx --input data/hellaswag_valid.jsonl --output output/hellaswag_valid_predict_davinci_002.jsonl
python davinci-002.py --api_key xxxxxxx --input data/socialiqa_valid.jsonl --output output/socialiqa_valid_predict_davinci_002.jsonl

# Vicuna
python get_response.py --model_path huggingface/vicuna-7b --test_file data/openbook_valid.jsonl --device cuda --answer_file output/openbook_valid_predict_vicuna.jsonl
python get_response.py --model_path huggingface/vicuna-7b --test_file data/hellaswag_valid.jsonl --device cuda --answer_file output/hellaswag_valid_predict_vicuna.jsonl
python get_response.py --model_path huggingface/vicuna-7b --test_file data/socialiqa_valid.jsonl --device cuda --answer_file output/socialiqa_valid_predict_vicuna.jsonl

# Alpaca
python get_response.py --model_path /media/public/models/huggingface/alpaca-7b --test_file data/openbook_valid.jsonl --device cuda --answer_file output/openbook_valid_predict_alpaca.jsonl
python get_response.py --model_path /media/public/models/huggingface/alpaca-7b --test_file data/hellaswag_valid.jsonl --device cuda --answer_file output/hellaswag_valid_predict_alpaca.jsonl
python get_response.py --model_path /media/public/models/huggingface/alpaca-7b --test_file data/socialiqa_valid.jsonl --device cuda --answer_file output/socialiqa_valid_predict_alpaca.jsonl

# ChatGLM
python get_response.py --model_path /media/public/models/huggingface/chatglm-6b --test_file data/openbook_valid.jsonl --device cuda --answer_file output/openbook_valid_predict_chatglm.jsonl
python get_response.py --model_path /media/public/models/huggingface/chatglm-6b --test_file data/hellaswag_valid.jsonl --device cuda --answer_file output/hellaswag_valid_predict_chatglm.jsonl
python get_response.py --model_path /media/public/models/huggingface/chatglm-6b --test_file data/socialiqa_valid.jsonl --device cuda --answer_file output/socialiqa_valid_predict_chatglm.jsonl

# LLaMA
python get_response.py --model_path /media/public/models/huggingface/llama-7b --test_file data/openbook_valid.jsonl --device cuda --answer_file output/openbook_valid_predict_llama.jsonl
python get_response.py --model_path /media/public/models/huggingface/llama-7b --test_file data/hellaswag_valid.jsonl --device cuda --answer_file output/hellaswag_valid_predict_llama.jsonl
python get_response.py --model_path /media/public/models/huggingface/llama-7b --test_file data/socialiqa_valid.jsonl --device cuda --answer_file output/socialiqa_valid_predict_llama.jsonl

# Falcon
python get_response.py --model_path /media/public/models/huggingface/falcon/falcon-7b --test_file data/openbook_valid.jsonl --device cuda --answer_file output/openbook_valid_predict_falcon.jsonl
python get_response.py --model_path /media/public/models/huggingface/falcon/falcon-7b --test_file data/hellaswag_valid.jsonl --device cuda --answer_file output/hellaswag_valid_predict_falcon.jsonl
python get_response.py --model_path /media/public/models/huggingface/falcon/falcon-7b --test_file data/socialiqa_valid.jsonl --device cuda --answer_file output/socialiqa_valid_predict_falcon.jsonl

# Pythia-12B
python get_response.py --model_path /media/public/models/huggingface/pythia-12b --test_file data/openbook_valid.jsonl --device cuda --answer_file output/openbook_valid_predict_pythia.jsonl
python get_response.py --model_path /media/public/models/huggingface/pythia-12b --test_file data/hellaswag_valid.jsonl --device cuda --answer_file output/hellaswag_valid_predict_pythia.jsonl
python get_response.py --model_path /media/public/models/huggingface/pythia-12b --test_file data/socialiqa_valid.jsonl --device cuda --answer_file output/socialiqa_valid_predict_pythia.jsonl

# Pythia-7B
python get_response.py --model_path /media/public/models/huggingface/pythia-6.9b --test_file data/openbook_valid.jsonl --device cuda --answer_file output/openbook_valid_predict_pythia_7b.jsonl
python get_response.py --model_path /media/public/models/huggingface/pythia-6.9b --test_file data/hellaswag_valid.jsonl --device cuda --answer_file output/hellaswag_valid_predict_pythia_7b.jsonl
python get_response.py --model_path /media/public/models/huggingface/pythia-6.9b --test_file data/socialiqa_valid.jsonl --device cuda --answer_file output/socialiqa_valid_predict_pythia_7b.jsonl