import time
import json
import random
import os
import slack
import argparse
import warnings
import anthropic

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
    print(claude_prompt)
    prompt = [claude_prompt, claude_prompt, claude_prompt]
    response = client.completion(
        prompt=claude_prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT, anthropic.AI_PROMPT],
        model=model,
        max_tokens_to_sample=max_tokens,
        temperature=0,
    )
    return response["completion"]
   

if __name__ == '__main__':
    result = call_claude_completion('Give me the result of 1 + 1')
    print(result)
    exit(0)