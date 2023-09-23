import time
import json
import random
import os
import slack
import argparse
import warnings
from flask import Flask
from flask_slack import Slack
from tqdm import tqdm 
import logging

import anthropic

client = anthropic.Client(
    "sk-ant-WqXB_mm8My-_VYnEqAj_p9XFtdyZ5-36a_YHABdTn_NabriJQI8k-J2_Ie3E9_pyfXdP_MEuXfXFbC5wt3LiOw"
)

# app = Flask(__name__)
# slack_app = Slack(app)

def get_exist_set(output_path):
    with open(output_path, "r") as f:
        all_data = f.readlines()
        all_data = [json.loads(line) for line in all_data]
        exist_id_set = [d['id'] for d in all_data]
    return set(exist_id_set)


def annotate(prompt):
    # prompt = "\n".join(prompt.split("\n")[:-1]) + "\n"
    claude_prompt = anthropic.HUMAN_PROMPT + prompt + anthropic.AI_PROMPT
    # claude_prompt = prompt
    response = client.completion(
        prompt=claude_prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT, anthropic.AI_PROMPT],
        model="claude-v1.3",
        max_tokens_to_sample=128,
        temperature=0,
    )
    return response["completion"].strip()

# def annotate(message):
#     # send to defined channel
#     client.chat_postMessage(channel=channel_id, text=message, as_user=True)

#     # get response
#     text = get_history()
#     temp = ''
#     while True:
#         temp = get_history()
#         if temp != text and 'Typing' not in temp:
#             break
#         else:
#             time.sleep(1)
#     response = temp.replace('\n\n', '\n').strip()
#     response = response.split('\n')[0]
#     print(response)
#     return response
   

if __name__ == '__main__':
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--userOAuthToken')
    parser.add_argument('--channel_id')
    parser.add_argument('--input')
    parser.add_argument('--output')
    
    args = parser.parse_args()
    
    # channel_id = args.channel_id
    # userOAuthToken = args.userOAuthToken
    # client = slack.WebClient(token=userOAuthToken)
    
    input = args.input
    output = args.output
    
    data_id2data = {}
    
    with open(input, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            each_data = json.loads(line)
            data_id2data[each_data['ID']] = each_data
                
    if os.path.exists(output):
        data_id_set = set(data_id2data.keys()) - get_exist_set(output)
        mode = "a+"
    else:
        data_id_set = set(data_id2data.keys())
        mode = "w"
    
    with open(output, mode) as w:
        for data_id in tqdm(data_id_set, total=len(data_id_set)):
            data = data_id2data[data_id]
        
            input = data['input']
            
            ans = ''
            while len(ans) == 0:
                try:
                    ans = annotate(input)
                except Exception as e:
                    logging.exception(e)
                    continue
            
            gen = {
                    "id": data['ID'],
                    "input": input,
                    "reference": data['ref'],
                    "generation": ans,
                }   
                
            w.write(json.dumps(gen) + "\n")