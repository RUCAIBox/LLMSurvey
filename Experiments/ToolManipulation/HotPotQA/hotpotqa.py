import argparse
import os

import openai
import requests

requests.packages.urllib3.disable_warnings()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="text-davinci-002", help="which model you want to use for eval")
    parser.add_argument("--api_key", type=str, default="", help="the api key provided for calling")
    args = parser.parse_args()

    openai.api_key = args.api_key
    model = args.model

    def change_api():
        api_key_list = ["",
                        "", ]
        if openai.api_key in api_key_list:
            api_key_list.remove(openai.api_key)
        k = random.randint(0, len(api_key_list) - 1)
        openai.api_key = api_key_list[k]
        print("\nUsing api_key: ", str(k), api_key_list[k])
        return k, api_key_list


    def llm(prompt, stop=["\n"]):
        while True:
            try:
                response = openai.Completion.create(
                    model= model,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=100,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=stop
                )
                response = response["choices"][0]["text"]
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
        # print(response)
        # print("============")
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
    webthink_simples = webthink_prompt.strip().split("Question:")[1:]
    webthink_simples = ["Question:"+_ for _ in webthink_simples]
    webthink_prompt = "\n".join(webthink_simples[0:3])
    instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
    (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
    (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
    (3) Finish[answer], which returns the answer and finishes the task.
    Here are some examples.
    """
    webthink_prompt = instruction + webthink_prompt

    def webthink(idx=None, prompt=webthink_prompt, to_print=True):
        question = env.reset(idx=idx)
        if to_print:
            print(idx, question)
        prompt += question + "\n"
        n_calls, n_badcalls = 0, 0
        for i in range(1, 8):
            n_calls += 1
            thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
            try:
                thought, action = thought_action.strip().split(f"\nAction {i}:")
                action = action.strip()
            except:
                print('ohh...', thought_action)
                n_badcalls += 1
                n_calls += 1
                thought = thought_action.strip().split('\n')[0]
                action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()

            obs, r, done, info = step(env, action[0].lower() + action[1:])
            obs = obs.replace('\\n', '')
            step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            prompt += step_str
            if to_print:
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