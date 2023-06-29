import os
import json
import logging
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from yacs.config import CfgNode
import IPython

from model.Model import Model

def save_benchmark(model, language, cache_path, responses):
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    file_path = os.path.join(cache_path, model + "_" + language + ".json")
    with open(file_path, "w", encoding="utf-8") as fo:
        json.dump(responses, fo)

class BBH3kBenchmark:

    def __init__(self, config: CfgNode, logger: logging.Logger) -> None:
        self.config: CfgNode = config
        self.logger: logging.Logger = logger
        # with open("data/bbh3k.json", "r", encoding="utf-8") as fi:
        with open("bbh/data/bbh_full.json", "r", encoding="utf-8") as fi:
        # with open("bbh/data/bbh_tiny.json", "r", encoding="utf-8") as fi:
            self.data = json.load(fi)
        self.cache_path = config.benchmark.cache_response_path
        self.save_response = config.benchmark.save_response
        self.language = config.benchmark.language

    def generate_text(self, model: Model, ipt: str):
        results = model.generate_text(ipt)
        return {
            "name": model.model_alias,
            "method": "generate_text",
            "msg": f"{results}"
        }

    def calc_acc(self, prompts: List[Dict], preds: List[str]) -> List[Dict]:
        def _calc_acc(question: str, gt: str, pred: str) -> float:
            question = question.lower()
            gt = gt.lower()
            pred=pred.split("\n")[0]
            pred = pred.lower()

            gts = [gt]

            if gt.find("(") != -1:
                start_index = question.find(gt)
                end_index = question.find("\n", start_index)
                gts.append(question[start_index + len(gt): end_index].lstrip())
                return float((gts[0] in pred) or (gts[1] in pred or pred in gts[1]))

            return float(gt in pred)
        questions=list(map(lambda prompt: prompt["question"], prompts))
        gts = list(map(lambda prompt: prompt["answer"], prompts))
        acc = list(map(lambda x: _calc_acc(*x), zip(questions,gts, preds)))
        return acc

    def evaluate_model(self, model: Model):
        preds = []
        responses=[]
        for prompt in tqdm(self.data,desc="Data"):
            # question="For the following questions please return only one word as an answer.\n" + prompt['question']
            question="For the following questions please return only one word as an answer.\n" + prompt['question'].rstrip('\nA: ')
            response = self.generate_text(model, question)
            self.logger.info("ques: " + question)
            self.logger.info("Model ans: " + json.dumps(response))
            self.logger.info("Correct ans: " + prompt["answer"])
            preds.append(response["msg"])
            prompt["response"]=response["msg"]
            responses.append(prompt)
        if self.save_response:
            save_benchmark(model.model_alias, self.language, self.cache_path, responses)
        acc = self.calc_acc(self.data, preds)

        task_acc = {}
        type_acc = {}
        task_cnt = {}
        type_cnt = {}

        for i in range(len(self.data)):
            task=self.data[i]["taskname"]
            type=self.data[i]["type"]
            if task not in task_cnt:
                task_cnt[task] = 0
                task_acc[task] = 0
            task_cnt[task] += 1
            task_acc[task] += acc[i]
            if type not in type_cnt:
                type_cnt[type]=0
                type_acc[type]=0
            type_cnt[type] += 1
            type_acc[type] += acc[i]

        task_result = pd.DataFrame()
        task_result.index=["accuracy"]
        sorted_keys = sorted(task_cnt.keys())
        task_accs = []
        for key in sorted_keys:
            avg_acc = task_acc[key] / task_cnt[key]
            task_result[key] = avg_acc
            task_accs.append(avg_acc)
            self.logger.info(f'The average accuracy of task {key}: {avg_acc}')
        self.logger.info("\n" + str(task_result))
        self.logger.info(f"Task average accuracy :{task_accs}")

        total_acc = 0
        type_result = pd.DataFrame()
        type_result.index = ["accuracy"]
        sorted_keys = sorted(type_cnt.keys())
        type_accs = []
        for key in sorted_keys:
            avg_acc = type_acc[key] / type_cnt[key]
            type_result[key] = avg_acc
            type_accs.append(avg_acc)
            total_acc += type_acc[key]
            self.logger.info(f'The average accuracy of type {key}: {avg_acc}')
        self.logger.info("\n" + str(type_result))
        avg_acc = total_acc / len(acc)
        self.logger.info(f"Type average accuracy :{type_accs}")
        self.logger.info(f'The global average accuracy: {avg_acc}')
