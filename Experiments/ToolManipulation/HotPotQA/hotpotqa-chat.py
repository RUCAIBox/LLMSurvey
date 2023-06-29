import argparse
import os
from copy import deepcopy

import anthropic
import openai
import requests

requests.packages.urllib3.disable_warnings()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="which model you want to use for eval")
    parser.add_argument("--api_key", type=str, default="", help="the api key provided for calling")
    args = parser.parse_args()

    openai.api_key = args.api_key
    model = args.model

    def change_api():
        api_key_list = ["",
                        "",]
        if openai.api_key in api_key_list:
            api_key_list.remove(openai.api_key)
        k = random.randint(0, len(api_key_list) - 1)
        openai.api_key = api_key_list[k]
        print("\nUsing api_key: ", str(k), api_key_list[k])
        return k, api_key_list


    def llm(prompt, stop=["\n"]):
        while True:
            try:
                responses = openai.ChatCompletion.create(
                    model=model,
                    messages=prompt,
                    temperature=0,
                    max_tokens=100,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=stop
                )
                response = responses['choices'][0]['message']['content']
                break
            except openai.error.RateLimitError as e:
                if str(e) == "You exceeded your current quota, please check your plan and billing details.":
                    change_api()
                    time.sleep(10)
                else:
                    print('\nopenai.error.RateLimitError\nRetrying...')
                    time.sleep(10)
            except openai.error.ServiceUnavailableError:
                print('\nopenai.error.ServiceUnavailableError\nRetrying...')
                time.sleep(10)
            except openai.error.Timeout:
                print('\nopenai.error.Timeout\nRetrying...')
                time.sleep(10)
            except openai.error.APIError:
                print('\nopenai.error.APIError\nRetrying...')
                time.sleep(10)
            except openai.error.APIConnectionError:
                print('\nopenai.error.APIConnectionError\nRetrying...')
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
    prompt_messages = []
    instruction_message = {"role": "user", "content": instruction}
    prompt_messages.append(instruction_message)

    for simple in webthink_simples:
        steps = simple.split("Thought")
        q_str = steps[0]
        q_message = {"role": "user", "content": q_str}
        prompt_messages.append(q_message)
        steps = ["Thought"+_ for _ in steps[1:]]
        for stp in steps[:-1]:
            ta, obs =  stp.split("Observation")
            ta_message = {"role": "assistant", "content": ta}
            prompt_messages.append(ta_message)
            obs_message = {"role": "user", "content": "Observation" + obs}
            prompt_messages.append(obs_message)

        finish_message = {"role": "assistant", "content": steps[-1]}
        prompt_messages.append(finish_message)

    def webthink(idx=None, prompt_messages=prompt_messages, to_print=True):
        prompt = deepcopy(prompt_messages)
        question = env.reset(idx=idx)
        if to_print:
            print(idx, question)
        q_message = {"role": "user", "content": question + "\n"}
        prompt.append(q_message)
        n_calls, n_badcalls = 0, 0
        for i in range(1, 8):
            n_calls += 1
            thought_action = llm(prompt, stop=[f"\nObservation {i}:"])
            try:
                thought, action = thought_action.strip().split(f"\nAction {i}: ")
                thought = thought.replace(f"Thought {i}:", "").strip()
            except:
                print('ohh...', thought_action)
                n_badcalls += 1
                n_calls += 1
                temp_prompt = deepcopy(prompt)
                thought = thought_action.replace(f"Thought {i}:","").strip().split('\n')[0]
                thought_message = {"role": "assistant", "content": f"Thought {i}: {thought}\n"}
                temp_prompt.append(thought_message)
                action = llm(temp_prompt, stop=[f"\n"]).strip().replace(f"Action {i}:", "").strip()
            obs, r, done, info = step(env, action[0].lower() + action[1:])
            obs = obs.replace('\\n', '')
            ta_message = {"role": "assistant", "content": f"Thought {i}: {thought}\nAction {i}: {action}\n"}
            prompt.append(ta_message)
            obs_message = {"role": "user", "content": f"Observation {i}: {obs}\n"}
            prompt.append(obs_message)
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
    for nn, i in enumerate(idxs[:500]):
        r, info = webthink(i, to_print=True)
        rs.append(info['em'])
        rs1.append(info['f1'])
        infos.append(info)
        print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
        print(sum(rs1), len(rs1), sum(rs1) / len(rs1), (time.time() - old_time) / len(rs1))
        print('-----------')
        print()