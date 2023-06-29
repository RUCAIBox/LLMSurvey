import argparse
import os
from copy import deepcopy

import anthropic
import requests

requests.packages.urllib3.disable_warnings()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="claude-v1.3", help="which model you want to use for eval")
    parser.add_argument("--api_key", type=str, default="", help="the api key provided for calling")
    args = parser.parse_args()

    api_key = args.api_key
    model = args.model

    def llm(prompt, stop=["\n"]):
        client = anthropic.Client(api_key)
        while True:
            try:
                responses = client.completion(
                    prompt=f"{prompt}{anthropic.AI_PROMPT}",
                    stop_sequences=[anthropic.HUMAN_PROMPT],
                    model=model,
                    max_tokens_to_sample=100,
                )
                response = responses["completion"].strip()
                break
            except anthropic.ApiException:
                print('\nanthropic.ApiException\nRetrying...')
                time.sleep(10)
            except requests.exceptions.SSLError as e:
                print('\nrequests.exceptions.SSLError\nRetrying...')
                time.sleep(10)
            except requests.exceptions.ProxyError as e:
                print('\nrequests.exceptions.ProxyError\nRetrying...')
                time.sleep(10)
            except ConnectionResetError as e:
                print('\nConnectionResetError\nRetrying...')
                time.sleep(10)
            except Exception as e:
                print(e)
                response = None
                break
        response = response.split(stop[0])[0]
        return response

    import wikienv, wrappers
    env = wikienv.WikiEnv()
    env = wrappers.HotPotQAWrapper(env, split="dev")
    env = wrappers.LoggingWrapper(env)

    def step(env, action):
        attempts = 0
        while attempts < 10:
            try:
                return env.step(action)
            except requests.exceptions.Timeout:
                attempts += 1

    import json
    import sys

    folder = './prompts/'
    prompt_file = 'prompts_naive.json'
    with open(folder + prompt_file, 'r') as f:
        prompt_dict = json.load(f)

    webthink_prompt = prompt_dict['webthink_simple6']
    webthink_simples = webthink_prompt.strip().split("Question:")[1:4]
    webthink_simples = ["Question:"+_ for _ in webthink_simples]

    instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
    (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
    (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
    (3) Finish[answer], which returns the answer and finishes the task.
    Here are some examples.
    """

    webthink_prompt = f"{anthropic.HUMAN_PROMPT} {instruction}"

    for simple in webthink_simples:
        steps = simple.split("Thought")
        q_str = steps[0]
        webthink_prompt += f"{anthropic.HUMAN_PROMPT} {q_str}"
        steps = ["Thought"+_ for _ in steps[1:]]
        for stp in steps[:-1]:
            ta, obs =  stp.split("Observation")
            webthink_prompt += f"{anthropic.AI_PROMPT} {ta}"
            obs="Observation" + obs
            webthink_prompt += f"{anthropic.HUMAN_PROMPT} {obs}"

        webthink_prompt += f"{anthropic.AI_PROMPT} {steps[-1]}"

    def webthink(idx=None, prompt=webthink_prompt, to_print=True):
        question = env.reset(idx=idx)
        if to_print:
            print(idx, question)
        prompt += f"{anthropic.HUMAN_PROMPT} {question}" + "\n"
        n_calls, n_badcalls = 0, 0
        for i in range(1, 8):
            n_calls += 1
            try:
                thought_action = llm(prompt, stop=[f"\nObservation {i}:"])
                try:
                    thought, action = thought_action.strip().split(f"\nAction {i}: ")
                    thought = thought.replace(f"Thought {i}:", "").strip()
                except:
                    # print('ohh...', thought_action)
                    n_badcalls += 1
                    n_calls += 1
                    temp_prompt = deepcopy(prompt)
                    thought = thought_action.replace(f"Thought {i}:", "").strip().split('\n')[0]
                    thought_message = f"Thought {i}: {thought}\n"
                    temp_prompt += f"{anthropic.AI_PROMPT} {thought_message}"
                    action = llm(temp_prompt, stop=[f"\n"]).strip().replace(f"Action {i}:", "").strip()
                obs, r, done, info = step(env, action[0].lower() + action[1:])
            except Exception as e:
                print(e)
                break
            obs = obs.replace('\\n', '')
            ta_message = f"Thought {i}: {thought}\nAction {i}: {action}\n"
            prompt+= f"{anthropic.AI_PROMPT} {ta_message}"
            obs_message = f"Observation {i}: {obs}\n"
            prompt+= f"{anthropic.HUMAN_PROMPT} {obs_message}"
            if to_print:
                step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
                print(step_str)
            if done:
                break
        if not done:
            obs, r, done, info = step(env, "finish[]")
        if to_print:
            print(info, '\n')
        info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
        return r, info

    import random
    import time
    idxs = list(range(7405))
    random.Random(233).shuffle(idxs)

    rs = []
    rs1 = []
    infos = []
    old_time = time.time()
    for i in idxs[:500]:
        r, info = webthink(i, to_print=True)
        rs.append(info['em'])
        rs1.append(info['f1'])
        infos.append(info)
        print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
        print(sum(rs1), len(rs1), sum(rs1) / len(rs1), (time.time() - old_time) / len(rs1))
        print('-----------')
        print()