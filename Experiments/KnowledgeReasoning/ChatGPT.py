import argparse
import copy
import json
import os
import re
import random
import time
import typing
import warnings
import tiktoken
from tqdm import tqdm
import openai
import nltk
from loguru import logger
from thefuzz import fuzz
from tenacity import Retrying, retry_if_not_exception_type, _utils
from tenacity.stop import stop_base
from tenacity.wait import wait_base

import sys
sys.path.append("..")
import logging
warnings.filterwarnings('ignore')

def get_exist_set(output_path):
    with open(output_path, "r") as f:
        all_data = f.readlines()
        all_data = [json.loads(line) for line in all_data]
        exist_id_set = [d['ID'] for d in all_data]
    return set(exist_id_set)

def my_before_sleep(retry_state):
    logger.debug(
        f'Retrying: attempt {retry_state.attempt_number} ended with: {retry_state.outcome}, spend {retry_state.seconds_since_start} in total')


class my_wait_exponential(wait_base):
    def __init__(
        self,
        multiplier: typing.Union[int, float] = 1,
        max: _utils.time_unit_type = _utils.MAX_WAIT,  # noqa
        exp_base: typing.Union[int, float] = 2,
        min: _utils.time_unit_type = 0,  # noqa
    ) -> None:
        self.multiplier = multiplier
        self.min = _utils.to_seconds(min)
        self.max = _utils.to_seconds(max)
        self.exp_base = exp_base

    def __call__(self, retry_state: "RetryCallState") -> float:
        if retry_state.outcome == openai.error.Timeout:
            return 0

        try:
            exp = self.exp_base ** (retry_state.attempt_number - 1)
            result = self.multiplier * exp
        except OverflowError:
            return self.max
        return max(max(0, self.min), min(result, self.max))


class my_stop_after_attempt(stop_base):
    """Stop when the previous attempt >= max_attempt."""

    def __init__(self, max_attempt_number: int) -> None:
        self.max_attempt_number = max_attempt_number

    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome == openai.error.Timeout:
            retry_state.attempt_number -= 1
        return retry_state.attempt_number >= self.max_attempt_number

def annotate(prompt, logit_bias=None):
    
    if logit_bias is None:
        logit_bias = {}

    request_timeout = 20
    while True:
        # with attempt:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo', messages=prompt, temperature=0, max_tokens=128,
                logit_bias=logit_bias,
                request_timeout=request_timeout,
            )['choices'][0]['message']['content']
            print(f"response={response}")
            break
        # except openai.error.RateLimitError as e:
        #     print("Retrying...", e)
        #     time.sleep(max(i + 1, 3))
        # except openai.InvalidRequestError as e:
        #     print("Retrying...", e)
        #     break
        except Exception as e:
            print("Retrying...", e)
            time.sleep(3)
            
        request_timeout = min(300, request_timeout * 2)

    return response


if __name__ == '__main__':
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key')
    parser.add_argument('--input')
    parser.add_argument('--output')
    
    args = parser.parse_args()
    openai.api_key = args.api_key
    input = args.input
    output = args.output
    
    data_id2data = {}
    
    with open(input, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            each_data = json.loads(line)
            data_id2data[each_data['ID']] = each_data
                
    # TODO: add dataset
    
    if os.path.exists(output):
        data_id_set = set(data_id2data.keys()) - get_exist_set(output)
        mode = "a+"
    else:
        data_id_set = set(data_id2data.keys())
        mode = "w"

    with open(output, 'w') as w:
        for data_id in tqdm(data_id_set, total=len(data_id_set)):
            data = data_id2data[data_id]
            

            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            logit_bias = {encoding.encode(option)[0]: 10 for option in ['A', 'B', 'C', 'D']}

            if 'openbook' in input:
                prompt_list = [{
                'role': 'system',
                'content': "You need to choose an answer to the question. You must only output A, B, C, or D."
                }]
            elif 'socialiqa' in input:
                prompt_list = [{
                'role': 'system',
                'content': "You need to choose an answer to the question. You must only output A, B, or C."
                }]
            elif 'hellaswag' in input:
                prompt_list = [{
                'role': 'system',
                'content': "You need to choose an answer to the question. You must only output A, B, C, or D."
                }]
            
            input = data['input']
            prompt_list = []
            
            prompt_list.append({
                'role': 'user',
                'content': input
            })
            
            
            ans = annotate(prompt_list, logit_bias).strip()
            gen = {
                    "id": data['ID'],
                    "input": input,
                    "reference": data['ref'],
                    "generation": ans,
                }   
            
            w.write(json.dumps(gen) + "\n")
