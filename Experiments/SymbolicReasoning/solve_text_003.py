import os
import openai
import json
import time
import random
import string
import argparse

from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Dataset

from data_process import (
    DataProcessForGSM8k,
    DataProcessForAQUA,
    DataProcessForMATH,
    DataProcessForPENGUINS,
    DataProcessForCOLOR,
)

# 账户的api调用密钥
openai.api_key = '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'

DATA_PROCESSER = {
    'gsm8k': DataProcessForGSM8k,
    'aqua_rat': DataProcessForAQUA,
    'math': DataProcessForMATH,
    'penguins': DataProcessForPENGUINS,
    'color': DataProcessForCOLOR,
}


def call_completion(prompt, stop_word='Problem'):
    while (True):
        try:
            res = openai.Completion.create(
                model='text-davinci-003',
                prompt=prompt,
                temperature=0,
                max_tokens=512,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_word
            )
            break
        except:
            time.sleep(20)
    
    step_list = []
    for choice in res['choices']:
        steps = choice['text'].strip()
        if "Problem" in steps:
            steps = steps.split('problem')[0].strip()
        if "Q: " in steps:
            steps = steps.split('Q: ')[0].strip()
        step_list.append(steps)
    return step_list


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


def main(args):
    fout = open(args.result_path, args.write_mode)
    data_processer = DATA_PROCESSER[args.dataset_name](args.demo_path, args.num_examplar)
    
    if ('gsm8k' in args.dataset_name):
        if (args.data_path is None):
            dataset = load_dataset(args.dataset_name, 'main', split='test')
        else:
            dataset = []
            with open(args.data_path, 'r') as fin:
                for data in load_multi_line_json(fin, 5):
                    dataset.append(data)
    elif ('math' in args.dataset_name):
        with open('dataset/math/test.json', 'r') as fin:
            raw_dataset = json.load(fin)
        dataset = {}
        for data in raw_dataset:
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
        dataset = load_dataset(args.dataset_name, split='test')
    print(dataset)

    num_correct = 0
    total_problem = 0
    batch = {
        'prompt': [],
        'data': []
    }

    def make_query(batch):
        nonlocal num_correct, total_problem
        llm_steps = call_completion(batch['prompt'])

        for llm_step, test_data in zip(llm_steps, batch['data']):
            data = test_data[0]
            real_label = test_data[1]

            data['llm_step'] = llm_step
            pred = llm_step.split('The answer is')[-1].strip()
            if (len(pred) >= 1 and pred[-1] == '.'):
                pred = pred[:-1]
            data['llm_answer'] = pred
            data['real_answer'] = real_label
            data['score'] = False

            # if (pred == real_label):
            if (pred in real_label):
                num_correct = num_correct + 1
                data['score'] = True
            total_problem = total_problem + 1

            fout.write(json.dumps(data, indent=4, ensure_ascii=False) + '\n')

    step = 0
    for data in tqdm(dataset):
        step = step + 1
        # if (step <= 3680): continue
        prompt, real_label = data_processer.process(data)
        batch['prompt'].append(prompt)
        batch['data'].append((data, real_label))

        if (len(batch['prompt']) == 10):
            make_query(batch)
            batch = {
                'prompt': [],
                'data': []
            }
    
    if (len(batch['prompt']) != 0):
        make_query(batch)

    print('Accuracy: {} ( {} / {} )'.format(round(num_correct / total_problem * 100, 2), num_correct, total_problem), file=fout)

    fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--write_mode', type=str, default='a', help='The mode to write result file')
    parser.add_argument('--result_path', type=str, help='The path to save result')
    parser.add_argument('--dataset_name', type=str, help='The name of dataset')
    parser.add_argument('--num_examplar', type=int, default=5, help='The number of examplar in prompt')
    parser.add_argument('--demo_path', type=str, help='The path to the demos')
    parser.add_argument('--data_path', type=str, default=None, help='The path to the dataset')
    
    args = parser.parse_args()

    main(args)
