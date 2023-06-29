import os
import openai
import json
import time
import random
import string
import argparse

from tqdm import tqdm
from datasets import load_dataset, Dataset

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

bootstrap = {
    "role": "system",
    "content": "You are an expert in mathematical problem. You should follow the example and answer the last question.",
}


def call_chat_completion(prompt, stop_word='Problem'):
    messages = [
        bootstrap,
        {"role": "user", "content": prompt},
    ]
    while (True):
        try:
            res = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=messages,
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

    choice = res['choices'][0]
    steps = choice['message']['content'].strip()
    if "Problem" in steps:
        steps = steps.split('problem')[0].strip()
    if "Q: " in steps:
        steps = steps.split('Q: ')[0].strip()
    return steps 


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
        with open('dataset/math/test_retrieval-classifier.json', 'r') as fin:
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
        # if (step <= 218): continue
        # for i in range(args.num_examplar):
        #     prompt, real_label = data_processer.process_retrieval(data, i, data['retrieval_result'])
        #     if (len(prompt) < int(4096 * 1.5)):
        #         break
        prompt, real_label = data_processer.process(data)
        if ('gsm8k' in args.dataset_name):
            if (args.prompt_type == 'problem'):
                prompt = 'You are an expert in mathematical problem. Here are some examples about mathematical problems. You can use the knowledge in examples and solve the last problem.\n' + prompt
            elif (args.prompt_type == 'answer'):
                prompt = 'You should generate the solution and use "The answer is" to express the final answer.\n' + prompt
        elif ('color' in args.dataset_name):
            if (args.prompt_type == 'problem'):
                prompt = 'You are an expert in reasoning problem. Here are some examples about symbolic reasoning. You can use the knowledge in examples and solve the last problem.\n' + prompt
            elif (args.prompt_type == 'answer'):
                prompt = 'You should follow the examples and generate the final answer without external solution or words.\n' + prompt
        llm_step = call_chat_completion(prompt)
        
        data['llm_step'] = llm_step
        pred = llm_step
        # if ('The answer is' in llm_step):
        #     pred = llm_step.split('The answer is')[-1].strip()
        # else:
        #     pred = get_answer_boxed(llm_step)
        # if (len(pred) >= 1 and pred[-1] == '.'):
        #     pred = pred[:-1]
        # if (len(pred) >= 2 and pred[0] == '$' and pred[-1] == '$'):
        #     pred = pred[1:-1]
        # data['prompt'] = prompt
        # pred = clean(pred)
        # real_label = clean(real_label)
        data['llm_answer'] = pred
        data['real_answer'] = real_label

        data['score'] = False
        # if (pred == real_label):
        if (pred in real_label):
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
    parser.add_argument('--prompt_type', type=str, default='None', help='The split of the dataset')
    
    args = parser.parse_args()

    main(args)
