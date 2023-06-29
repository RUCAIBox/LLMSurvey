cd XSum
mkdir generation
python xsum_chatgpt.py
python get_rougel.py --test_file generation/gpt-3.5-turbo.json
python xsum_003.py
python get_rougel.py --test_file generation/text-davinci-003.json
python xsum_002.py
python get_rougel.py --test_file generation/text-davinci-002.json
python xsum_claude.py --userOAuthToken {your_token} --channel_id {your_channelid}
python get_rougel.py --test_file generation/Claude.json
python open-source_model.py --model-path {vicuna7b_model_path} --test-file data/xsum.json --device cuda:0
python get_rougel.py --test_file generation/vicuna-7b.json
python open-source_model.py --model-path {alpaca7b_model_path} --test-file data/xsum.json --device cuda:0
python get_rougel.py --test_file generation/alpaca-7b.json
python open-source_model.py --model-path {chatglm6b_model_path} --test-file data/xsum.json --device cuda:0
python get_rougel.py --test_file generation/chatglm-6b.json
python open-source_model.py --model-path {llama7b_model_path} --test-file data/xsum.json --device cuda:0
python get_rougel.py --test_file generation/llama-7b.json
python open-source_model.py --model-path {falcon7b_model_path} --test-file data/xsum.json --device cuda:0
python get_rougel.py --test_file generation/falcon-7b.json
python open-source_model.py --model-path {pythia12b_model_path} --test-file data/xsum.json --device cuda:0
python get_rougel.py --test_file generation/pythia-12b.json
python open-source_model.py --model-path {pythia6.9b_model_path} --test-file data/xsum.json --device cuda:0
python get_rougel.py --test_file generation/pythia-6.9b.json
cd ..
