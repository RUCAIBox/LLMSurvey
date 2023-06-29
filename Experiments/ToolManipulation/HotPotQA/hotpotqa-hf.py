import argparse
import os

import requests
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

requests.packages.urllib3.disable_warnings()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    model_path = args.model_path
    device = args.device

    def load_model(
            model_path: str,
            device: str,
    ):
        if device == "cpu":
            kwargs = {"torch_dtype": torch.float32}
        elif "cuda" in device:
            kwargs = {"torch_dtype": torch.float16}
        else:
            raise ValueError(f"Invalid device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **kwargs,
            ).to(device)
        except ValueError:
            model = AutoModel.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **kwargs,
            ).to(device)
        except Exception as e:
            print(e)
            return None

        return model, tokenizer


    model, tokenizer = load_model(model_path,device)


    def llm(prompt, stop=["\n"]):
        inputs = tokenizer([prompt])
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        # print(len(input_ids),len(attention_mask))
        output_ids = model.generate(
            input_ids=torch.as_tensor(input_ids).to(device),
            attention_mask=torch.as_tensor(attention_mask).to(device),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=100,
            early_stopping=True,
            # eos_token_id=tokenizer.eos_token_id,
        )
        output_ids = output_ids[0][len(input_ids[0]):]
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        response = response.strip().split(stop[0])[0]
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
            try:
                thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
                try:
                    thought, action = thought_action.strip().split(f"\nAction {i}: ")
                except:
                    # print('ohh...', thought_action)
                    n_badcalls += 1
                    n_calls += 1
                    thought = thought_action.strip().split('\n')[0]
                    action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
                obs, r, done, info = step(env, action[0].lower() + action[1:])
            except Exception as e:
                print(e)
                done = False
                break
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
    for i in idxs[:500]:
        r, info = webthink(i, to_print=True)
        rs.append(info['em'])
        rs1.append(info['f1'])
        infos.append(info)
        print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
        print(sum(rs1), len(rs1), sum(rs1) / len(rs1), (time.time() - old_time) / len(rs1))
        print('-----------')
        print()