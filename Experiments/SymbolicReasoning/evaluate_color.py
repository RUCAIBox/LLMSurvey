import re
import json
import argparse

import sympy
from sympy.parsing.latex import parse_latex

import timeout_decorator

from tqdm import tqdm


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


def extract_answer(pred, label):
    pred = pred.split('Solution: ')[4].strip()
    pred = pred.split('Problem: ')[0].strip()
    pred = pred.split('\n')[0].strip()
    
    return (pred in label)


def main(args):
    with open(args.result_path, 'r') as fin:
        datas = load_multi_line_json(fin)
    
    num_correct = 0
    total_problem = 0
    for data in tqdm(datas):
        try:
            if (extract_answer(data['pred_ans'], data['real_ans']) == True):
                num_correct = num_correct + 1
        except:
            pass
        total_problem = total_problem + 1

    print('Accuracy: {} ( {} / {} )'.format(round(num_correct / total_problem * 100, 2), num_correct, total_problem))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_path', type=str, help='The path to result')
    
    args = parser.parse_args()

    main(args)