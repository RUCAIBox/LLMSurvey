import os
import re
import json
import torch
import random
import argparse
import transformers
from tqdm import tqdm
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from prompt_pattern import PROMPT, STOP_WORD

def clean(content):
    pattern = '<<.+>>'
    result = re.findall(pattern, content)
    for t in result:
        content = content.replace(t, '')
    content = content.replace('\n', '. ')
    return content

def load_multi_line_json(f):
    data = ''
    all_data = []
    raw_data =f.readlines()
    for line in raw_data:
        data = data + line
        if (line.startswith('}')):
            all_data.append(json.loads(data))
            data = ''
    return all_data

def get_answer_boxed(content):
    pattern = '\\boxed'
    start_pos = content.rfind(pattern)
    if (start_pos == -1): return content
    answer = ''
    num_left = 0
    for i in range(start_pos + 7, len(content)):
        if (content[i] == '}' and num_left == 0):
            break
        if (content[i] == '{'):
            num_left = num_left + 1
        elif (content[i] == '}'):
            num_left = num_left - 1
        answer = answer + content[i]
    return answer

parser = argparse.ArgumentParser()

parser.add_argument('--start_idx', type=int, default=0, help='The mode to write result file')
parser.add_argument('--end_idx', type=int, default=5000, help='The path to save result')

args = parser.parse_args()

print(args.start_idx, args.end_idx)

model = "/media/public/models/huggingface/falcon/falcon-7b"

set_seed(2023)
random.seed(2023)

pattern = PROMPT['VANILLA']
stop = STOP_WORD['VANILLA']

train_dataset = DatasetDict.from_json('/mnt/chenzhipeng/llm_data/pretrain_data/MATH/train.json')
test_dataset = DatasetDict.from_json('/mnt/chenzhipeng/llm_data/pretrain_data/MATH/test.json')

ids = random.sample([i for i in range(len(train_dataset))], 3)
demo = ''

for idx in ids:
    data = train_dataset[idx]
    problem = data['problem']
    solution = data['solution']
    answer = get_answer_boxed(solution)
    demo = demo + pattern.format(problem, f'{solution} The answer is {answer}')
print(demo)
print(test_dataset)
print(len(demo))

batch_size = 6

tokenizer = AutoTokenizer.from_pretrained(model, padding_side='left')

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    batch_size=batch_size
)
pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

fout = open('result/falcon_7b-math_pipeline/falcon_7b-math_pipeline-{}_{}-3shot.json'.format(args.start_idx, args.end_idx), 'w')

inputs = []
origin_data = []

def make_query():
    global inputs, origin_data, fout
    sequences = pipeline(
        inputs,
        max_length=1500,
        do_sample=False,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    for pred, data in zip(sequences, origin_data):
        tmp_data = data
        tmp_data['real_ans'] = get_answer_boxed(data['solution'])
        tmp_data['pred_ans'] = pred[0]['generated_text']
        fout.write(json.dumps(tmp_data, indent=4) + '\n')
    
    origin_data = []
    inputs = []

step = 0
for data in tqdm(test_dataset):
    if (step < args.start_idx): 
        step = step + 1
        continue
    if (step >= args.end_idx): 
        step = step + 1
        continue
    step = step + 1

    inputs.append(demo + 'Problem: ' + data['problem'] + "Solution: Let's think step by step")
    origin_data.append(data)
    if (len(inputs) == batch_size):
        make_query()
if (len(inputs) != 0):
    make_query()

fout.close()