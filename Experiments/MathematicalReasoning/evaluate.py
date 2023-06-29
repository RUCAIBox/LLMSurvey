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


def get_answer_boxed(content):
    pattern = '\\boxed'
    start_pos = content.rfind(pattern)
    if (start_pos == -1): return None
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


def del_answer_text(content):
    pattern = '\\text'
    start_pos = content.rfind(pattern)
    if (start_pos == -1): return content
    answer = ''
    num_left = 0
    for i in range(start_pos + 6, len(content)):
        if (content[i] == '}' and num_left == 0):
            if (i + 1 < len(content)):
                answer = answer + content[i + 1:]
            break
        if (content[i] == '{'):
            num_left = num_left + 1
        elif (content[i] == '}'):
            num_left = num_left - 1
        answer = answer + content[i]
    if (start_pos > 0):
        answer = content[:start_pos] + answer
    return answer

@timeout_decorator.timeout(2)
def test_equal(equ1, equ2):
    try:
        if (equ1.equals(equ2) == True):
            return True
    except:
        pass
    try:
        val_equ1 = sympy.N(equ1)
        val_equ2 = sympy.N(equ2)
        if (abs(float(val_equ1) - float(val_equ2)) <= 0.0001):
            return True
    except:
        pass
    return False

@timeout_decorator.timeout(2)
def extract_answer2(Answer_Texts, label):
    if ('\\boxed' in Answer_Texts):
        Answer_Texts = get_answer_boxed(Answer_Texts)
    if ('theansweris' in Answer_Texts.lower()):
        Answer_Texts = Answer_Texts.lower().split('theansweris')[-1].strip()
    
    Answer_Texts = del_answer_text(Answer_Texts)
    label = del_answer_text(label)

    # xxx(round to near ...)
    round_pattern = '\(roundedto.*?\)'
    round_results = re.findall(round_pattern, Answer_Texts)
    for rr in round_results:
        if (rr in Answer_Texts):
            Answer_Texts = Answer_Texts.lower().split(rr)[0].strip()
    
    # approximately xxx
    if ('approximately' in Answer_Texts):
        Answer_Texts = Answer_Texts[Answer_Texts.rfind('approximately') + 1:].strip()
        # Answer_Texts = Answer_Texts.split('approximately')[-1].strip()
    
    # xxx meters
    units = ['meter', 'kilometer', 'kilogram', 'degree', '^\\circ', 'square', 'inches', 'squareunits', 'cm', 'km', 'pound', 'mph', 'hours', 'dollar']
    for unit in units:
        if (unit in Answer_Texts):
            Answer_Texts = Answer_Texts.split(unit)[0].strip()
        if (unit in label):
            label = label.split(unit)[0].strip()
    
    #move un-related symbols
    Answer_Texts = Answer_Texts.replace('\"', '')
    Answer_Texts = Answer_Texts.replace('\'', '')
    label = label.replace('\"', '')
    label = label.replace('\'', '')

    Answer_Texts = Answer_Texts.replace('\\%', '/ 100')
    label = label.replace('\\%', '/ 100')
    if ('$or$' in Answer_Texts):
        first_part = Answer_Texts.split('$or$')[0].strip()
        second_part = Answer_Texts.split('$or$')[-1].strip()
        try:
            sp_first = parse_latex(first_part)
            sp_second = parse_latex(second_part)
            sp_label = parse_latex(label)
            if (test_equal(sp_first, sp_second) == True and test_equal(sp_first, sp_label) == True):
                return True
        except:
            pass

    Answer_Texts = Answer_Texts.replace('$', '')
    label = label.replace('$', '')

    try:
        sp_ans = parse_latex(Answer_Texts)
        sp_label = parse_latex(label)
        if (test_equal(sp_ans, sp_label) == True):
            return True
    except:
        pass

    Answer_Texts = Answer_Texts.replace('\\', '')
    Answer_Texts = Answer_Texts.replace(',!', '')
    Answer_Texts = Answer_Texts.replace(',', '')
    label = label.replace('\\', '')
    label = label.replace(',!', '')
    label = label.replace(',', '')
    
    if len(Answer_Texts) > 0:
        if '.' == Answer_Texts[-1]:
            Answer_Texts = Answer_Texts[:-1]
    Answer_Texts = Answer_Texts.replace(' ', '')

    #make 'dfrac'='frac'
    label = label.replace('dfrac', 'frac')
    Answer_Texts = Answer_Texts.replace('dfrac', 'frac')
    
    if Answer_Texts.rfind('=') > 0:
        Answer_Texts = Answer_Texts[Answer_Texts.rfind('=') + 1:]
    
    #1.00 is not equal to 1 problem
    try:
        if float(int(float(Answer_Texts))) - float(Answer_Texts) == 0:
            Answer_Texts = str(int(float(Answer_Texts)))
    except:
        Answer_Texts = Answer_Texts
    
    try:
        if abs(float(Answer_Texts) - float(label)) <= 0.0001:
            Answer_Texts = label
    except:
        Answer_Texts = Answer_Texts
    
    #make -{a}/{b}={-a}/{b}
    def move_reduce_sign(text):
        index=text.find('-')
        if index>=0:
            return '-'+text[:index]+text[index+1:]
        else:
            return text
    def find_nominator(text):
        index=text.find('{')
        index2=text.find('}')
        return text[index+1:index2]
    def find_denominator(text):
        index=text.rfind('{')
        index2=text.rfind('}')
        return text[index+1:index2]

    if 'frac' in Answer_Texts:
        Answer_Texts=move_reduce_sign(Answer_Texts)
        label=move_reduce_sign(label)

    # a cdot b -> ab
    if label.find('cdot')>=0:
        if Answer_Texts.find('cdot')<0:
            label=label.replace('\\cdot','')
    answer_state = True

    if Answer_Texts != label:
        answer_state = False
    # solving {a*b}/{a*c}!={b}/{c} question by turn the fraction into decimal.
    if label.find('\\dfrac')==0:
        try:
            label_float = float(find_nominator(label)) / float(find_denominator(label))
        except:
            label_float = 'Label can not convert to decimal'
        if Answer_Texts.find('\\dfrac')==0:
            try:
                answer_float = float(find_nominator(Answer_Texts)) / float(find_denominator(Answer_Texts))
            except:
                answer_float = 'Answer can not convert to decimal'
        else:
            try:
                #exec('answer_float=Answer_Texts')
                answer_float=float(Answer_Texts)
            except:
                answer_float='Answer can not convert to decimal'

        if answer_float==label_float:
                answer_state=True
    if Answer_Texts.find('\\dfrac')==0:
        try:
            answer_float = float(find_nominator(Answer_Texts)) / float(find_denominator(Answer_Texts))
        except:
            answer_float = 'Answer can not convert to decimal'
        if label.find('\\dfrac')==0:
            try:
                label_float=float(find_nominator(label))/float(find_denominator(label))
            except:
                label_float='Label can not convert to decimal'
        else:
            try:
                label_float = float(label)
            except:
                label_float = 'Label can not convert to decimal'
        if answer_float==label_float:
            answer_state=True
    return answer_state


def extract_answer(pred, label):
    pred = pred.split('\n')[0].strip()
    return pred == label


def main(args):
    with open(args.result_path, 'r') as fin:
        datas = load_multi_line_json(fin)
    
    num_correct = 0
    total_problem = 0
    for data in tqdm(datas):
        try:
            if (extract_answer2(data['pred_ans'], data['real_ans']) == True):
                num_correct = num_correct + 1
                # print(data['pred_ans'], data['real_ans'])
        except:
            pass
        total_problem = total_problem + 1

    print('Accuracy: {} ( {} / {} )'.format(round(num_correct / total_problem * 100, 2), num_correct, total_problem))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_path', type=str, help='The path to result')
    
    args = parser.parse_args()

    main(args)