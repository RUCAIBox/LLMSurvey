import os
import openai
import json
import time
import random
import string
import argparse
import anthropic

from tqdm import tqdm
from datasets import load_dataset, Dataset

from data_process import (
    DataProcessForGSM8k,
    DataProcessForAQUA,
    DataProcessForMATH,
    DataProcessForPENGUINS,
    DataProcessForCOLOR,
)

DATA_PROCESSER = {
    'gsm8k': DataProcessForGSM8k,
    'aqua_rat': DataProcessForAQUA,
    'math': DataProcessForMATH,
    'penguins': DataProcessForPENGUINS,
    'color': DataProcessForCOLOR,
}

bootstrap = {
    "role": "system",
    "content": "You are an expert in mathematical problem. You should follow the example and answer the last question.",
}


client = anthropic.Client(
    "sk-ant-WqXB_mm8My-_VYnEqAj_p9XFtdyZ5-36a_YHABdTn_NabriJQI8k-J2_Ie3E9_pyfXdP_MEuXfXFbC5wt3LiOw"
)

def call_claude_completion(
    prompt,
    model="claude-instant-v1",
    stop=None,
    max_tokens=512,
):
    claude_prompt = anthropic.HUMAN_PROMPT + prompt + anthropic.AI_PROMPT
    response = client.completion(
        prompt=claude_prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT, anthropic.AI_PROMPT],
        model=model,
        max_tokens_to_sample=max_tokens,
        temperature=0,
    )
    return response["completion"].strip()


def load_multi_line_json(f, num_line=13):
    data = ''
    while True:
        data = ''
        try:
            for _ in range(num_line):
                data += f.readline()
            yield json.loads(data)
        except:
            break


def clean(content):
    content = content.replace(' ', '')
    return content


def get_answer_boxed(content):
        pattern = '\\boxed'
        start_pos = content.rfind(pattern)
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


def main(args):
    fout = open(args.result_path, args.write_mode)
    data_processer = DATA_PROCESSER[args.dataset_name](args.demo_path, args.num_examplar)
    
    if ('gsm8k' in args.dataset_name):
        dataset = load_dataset(args.dataset_name, 'main', split=args.data_split)
    elif ('math' == args.dataset_name):
        with open('/mnt/chenzhipeng/llm_data/pretrain_data/MATH/test.json', 'r') as fin:
            raw_dataset = json.load(fin)
        dataset = {}
        for data in raw_dataset:
            if (args.data_split != 'test' and data['knowledge_point'] != args.data_split):
                continue
            for key in data.keys():
                if (key not in dataset):
                    dataset[key] = []
                dataset[key].append(data[key])
        dataset = Dataset.from_dict(dataset)
    elif ('penguins' in args.dataset_name):
        with open('dataset/penguins/test.json', 'r') as fin:
            raw_dataset = json.load(fin)
        dataset = {
            'problem': [],
            'solution': []
        }
        prefix = raw_dataset['task_prefix'].strip() + '\n'
        for data in raw_dataset['examples']:
            dataset['problem'].append('Problem: ' + prefix + data['input'] + '\nAnswer:')
            dataset['solution'].append(data['target'])
        dataset = Dataset.from_dict(dataset)
    elif ('color' in args.dataset_name):
        with open('dataset/colored_objects/test.json', 'r') as fin:
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
        dataset = Dataset.from_dict(dataset)
    else:
        dataset = load_dataset(args.dataset_name, split=args.data_split)
    print(dataset)

    num_correct = 0
    total_problem = 0
    step = 0
    for data in tqdm(dataset):
        step = step + 1
        # if (step <= 3747): continue
        # for i in range(args.num_examplar):
        #     prompt, real_label = data_processer.process(data, i)
        #     if (len(prompt) < int(4096 * 1.5)):
        #         break
        prompt, real_label = data_processer.process(data)
        llm_step = call_claude_completion(prompt)
        
        data['llm_step'] = llm_step
        pred = llm_step.lower().split('<ans>')[-1].strip()
        pred = pred.split('</ans>')[0].strip()

        if (len(pred) >= 1 and pred[-1] == '.'):
            pred = pred[:-1]
        if (len(pred) >= 2 and pred[0] == '$' and pred[-1] == '$'):
            pred = pred[1:-1]
        data['prompt'] = prompt
        # pred = clean(pred)
        # real_label = clean(real_label)
        data['llm_answer'] = pred
        data['real_answer'] = real_label

        data['score'] = False
        if (pred == real_label):
        # if (pred in real_label):
            num_correct = num_correct + 1
            data['score'] = True
        total_problem = total_problem + 1

        fout.write(json.dumps(data, indent=4, ensure_ascii=False) + '\n')

    print('Accuracy: {} ( {} / {} )'.format(round(num_correct / total_problem * 100, 2), num_correct, total_problem), file=fout)

    fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--write_mode', type=str, default='a', help='The mode to write result file')
    parser.add_argument('--result_path', type=str, help='The path to save result')
    parser.add_argument('--dataset_name', type=str, help='The name of dataset')
    parser.add_argument('--num_examplar', type=int, default=5, help='The number of examplar in prompt')
    parser.add_argument('--demo_path', type=str, help='The path to the demos')
    parser.add_argument('--data_split', type=str, default='test', help='The split of the dataset')
    
    args = parser.parse_args()

    main(args)
