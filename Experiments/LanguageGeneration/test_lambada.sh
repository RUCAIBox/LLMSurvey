cd LAMBADA
mkdir generation
python lambada_chatgpt.py
python lambada_003.py
python lambada_002.py
python lambada_claude.py --userOAuthToken {your_token} --channel_id {your_channelid}
python open-source_model.py --model-path {vicuna7b_model_path} --test-file data/halueval.json --device cuda:0
python open-source_model.py --model-path {alpaca7b_model_path} --test-file data/halueval.json --device cuda:0
python open-source_model.py --model-path {chatglm6b_model_path} --test-file data/halueval.json --device cuda:0
python open-source_model.py --model-path {llama7b_model_path} --test-file data/halueval.json --device cuda:0
python open-source_model.py --model-path {falcon7b_model_path} --test-file data/halueval.json --device cuda:0
python open-source_model.py --model-path {pythia12b_model_path} --test-file data/halueval.json --device cuda:0
python open-source_model.py --model-path {pythia6.9b_model_path} --test-file data/halueval.json --device cuda:0
cd ..
