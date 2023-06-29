## Datasets

 The paths of these two datasets:

```
Gorilla: ./Gorilla/eval/eval-data/questions/
HotPotQA: ./HotPotQA/data/
```

## Requirements

```
torch
transformers
openai
anthropic
tree_sitter
```

Note that Falcon LLMs require PyTorch 2.0 for use with `transformers`!

## Run

### Gorilla

#### 1. Getting Responses

LLM APIs:

```shell
cd ./Gorilla/eval
python get_llm_responses.py --model gpt-3.5-turbo --api_key $API_KEY --output_file gpt-3.5-turbo_torchhub_0_shot.jsonl --question_data eval-data/questions/torchhub/questions_torchhub_0_shot.jsonl --api_name torchhub
```

Open source models:

```shell
cd ./Gorilla/eval
python get_hf_responses.py --model_path $MODEL_PATH --question_file eval-data/questions/torchhub/questions_torchhub_0_shot.jsonl --device cuda --answer_file torchhub_0_shot.jsonl --api_name torchhub
```

#### 2. Evaluate the Response

```shell
cd ./Gorilla/eval/eval-scripts
python ast_eval_th.py --api_dataset ../../data/api/torchhub_api.jsonl --apibench ../../data/apibench/torchhub_eval.json --llm_responses gpt-3.5-turbo_torchhub_0_shot.jsonl
```

### HotPotQA

#### 1. LLM APIs

text-davinci-002 or text-davinci-003:

```shell
cd ./HotPotQA
python hotpotqa.py --model text-davinci-002 --api_key $API_KEY
```

gpt-3.5-turbo:

```shell
cd ./HotPotQA
python hotpotqa-chat.py --model gpt-3.5-turbo --api_key $API_KEY
```

claude:

```shell
cd ./HotPotQA
python hotpotqa-claude.py --model claude-v1.3 --api_key $API_KEY
```

#### 2. Open source models

```shell
cd ./HotPotQA
python hotpotqa-hf.py --model-path $MODEL_PATH --device cuda
```



