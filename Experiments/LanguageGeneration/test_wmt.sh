cd WMT22
mkdir generation
python wmt_chatgpt.py
python test_bleu.py --test_file generation/gpt-3.5-turbo.json
python wmt_003.py
python test_bleu.py --test_file generation/text-davinci-003.json
python wmt_002.py
python test_bleu.py --test_file generation/text-davinci-002.json
python wmt_claude.py --userOAuthToken {your_token} --channel_id {your_channelid}
python test_bleu.py --test_file generation/Claude.json
python open-source_model.py --model-path {vicuna7b_model_path} --test-file data/wmt.json --device cuda:0
python test_bleu.py --test_file generation/vicuna-7b.json
python open-source_model.py --model-path {alpaca7b_model_path} --test-file data/wmt.json --device cuda:0
python test_bleu.py --test_file generation/alpaca-7b.json
python open-source_model.py --model-path {chatglm6b_model_path} --test-file data/wmt.json --device cuda:0
python test_bleu.py --test_file generation/chatglm-6b.json
python open-source_model.py --model-path {llama7b_model_path} --test-file data/wmt.json --device cuda:0
python test_bleu.py --test_file generation/llama-7b.json
python open-source_model.py --model-path {falcon7b_model_path} --test-file data/wmt.json --device cuda:0
python test_bleu.py --test_file generation/falcon-7b.json
python open-source_model.py --model-path {pythia12b_model_path} --test-file data/wmt.json --device cuda:0
python test_bleu.py --test_file generation/pythia-12b.json
python open-source_model.py --model-path {pythia6.9b_model_path} --test-file data/wmt.json --device cuda:0
python test_bleu.py --test_file generation/pythia-6.9b.json
cd ..
