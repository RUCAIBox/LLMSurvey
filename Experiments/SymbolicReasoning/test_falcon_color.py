import os
import re
import json
import torch
import random
import transformers
from tqdm import tqdm
from datasets import DatasetDict, Dataset
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

model = "model/falcon/"

set_seed(2023)
random.seed(2023)

pattern = PROMPT['VANILLA']
stop = STOP_WORD['VANILLA']

def load_from_json(folder_path):
    demo_file = os.path.join(folder_path, 'demo.txt')
    with open(demo_file, 'r') as fin:
        demo = fin.readlines()
    demo = ''.join(demo)
    demo = demo.strip() + '\n\n'

    data_file = os.path.join(folder_path, 'test.json')
    with open(data_file, 'r') as fin:
        raw_dataset = json.load(fin)
    dataset = {
        'problem': [],
        'solution': []
    }
    for data in raw_dataset['examples']:
        dataset['problem'].append('Problem: ' + data['input'] + '\nAnswer:')
        tmp_ans = []
        for k, v in data['target_scores'].items():
            if (v == 1):
                tmp_ans.append(k)
        assert(len(tmp_ans) > 0)
        dataset['solution'].append(tmp_ans)
    return demo, Dataset.from_dict(dataset)

demo, test_dataset = load_from_json('dataset/colored_objects')
print(test_dataset)
print(demo)

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

fout = open('result/falcon-color-3shot.json', 'w')

inputs = []
origin_data = []

def make_query():
    global inputs, origin_data, fout
    sequences = pipeline(
        inputs,
        max_length=450,
        do_sample=False,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    for pred, data in zip(sequences, origin_data):
        tmp_data = data
        tmp_data['real_ans'] = data['solution']
        tmp_data['pred_ans'] = pred[0]['generated_text']
        fout.write(json.dumps(tmp_data, indent=4) + '\n')
    
    origin_data = []
    inputs = []

for step, data in enumerate(tqdm(test_dataset)):
    inputs.append(demo + data['problem'])
    origin_data.append(data)
    if (len(inputs) == batch_size):
        make_query()
if (len(inputs) != 0):
    make_query()

fout.close()