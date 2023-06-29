import re
import json
import torch
import random
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

model = "/media/public/models/huggingface/falcon/falcon-7b"

set_seed(2023)
random.seed(2023)

pattern = PROMPT['VANILLA']
stop = STOP_WORD['VANILLA']

train_dataset = DatasetDict.from_json('/mnt/chenzhipeng/llm_data/pretrain_data/MathInstruction/gsm8k.json', field='train')
ids = random.sample([i for i in range(len(train_dataset))], 3)
demo = ''
for idx in ids:
    data = train_dataset[idx]
    problem = data['question']
    solution = data['answer']
    answer = solution.split('####')[-1].strip()
    solution = clean(solution.split('####')[0].strip())
    demo = demo + pattern.format(problem, f'{solution} The answer is {answer}')

print(demo)

test_dataset = DatasetDict.from_json('/mnt/chenzhipeng/llm_data/pretrain_data/MathInstruction/gsm8k.json', field='test')
print(test_dataset)

eval_problem = []
test_data = []
with open('result/falcon_7b-gsm8k_pipeline-3shot.json') as fin:
    eval_dataset = load_multi_line_json(fin)
    for data in eval_dataset:
        eval_problem.append(data['question'])
    for data in test_dataset:
        if (data['question'] not in eval_problem):
            test_data.append(data)
            print(data['question'])

test_dataset = test_data

tokenizer = AutoTokenizer.from_pretrained(model, padding_side='left')
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    batch_size=6
)
pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

fout = open('result/falcon_7b-gsm8k_pipeline-3shot.json', 'a')

inputs = []
origin_data = []

def make_query():
    global inputs, origin_data, fout
    sequences = pipeline(
        inputs,
        max_length=1024,
        do_sample=False,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    for pred, data in zip(sequences, origin_data):
        tmp_data = data
        tmp_data['real_ans'] = data['answer'].split('####')[-1].strip()
        tmp_data['pred_ans'] = pred[0]['generated_text']
        fout.write(json.dumps(tmp_data, indent=4) + '\n')
    
    origin_data = []
    inputs = []

for step, data in enumerate(tqdm(test_dataset)):
    inputs.append(demo + 'Problem: ' + data['question'] + "Solution: Let's think step by step")
    origin_data.append(data)
    if (len(inputs) == 6):
        make_query()
if (len(inputs) != 0):
    make_query()

fout.close()