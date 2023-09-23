import json
import argparse
import os
import warnings
import logging

from tqdm import tqdm
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTNeoXTokenizerFast
)


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

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,use_fast=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **kwargs,
        )
    except ValueError:
        model = AutoModel.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **kwargs,
        )
    except Exception as e:
        logging.exception(e)
        return None

    return model, tokenizer

def get_exist_set(output_path):
    with open(output_path, "r") as f:
        all_data = f.readlines()
        all_data = [json.loads(line) for line in all_data]
        exist_id_set = [d['id'] for d in all_data]
    print("Already processed: ", exist_id_set)
    return set(exist_id_set)

def get_input(input, output):
    data_id2data = {}
    
    with open(input, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            each_data = json.loads(line)
            data_id2data[each_data['ID']] = each_data
    print("Totally %d test samples"%len(data_id2data))
                
    # TODO: add dataset
    
    if os.path.exists(output):
        data_id_set = set(data_id2data.keys()) - get_exist_set(output)
        mode = "a+"
    else:
        data_id_set = set(data_id2data.keys())
        mode = "w"
    print("Totally %d test samples need to process"%len(data_id_set))

    # Load questions file
    # question_jsons = []
    # if question_file.endswith("jsonl"):
    #     with open(question_file, "r") as ques_file:
    #         all_lines = ques_file.readlines()
    #         for line in all_lines:
                # question_jsons.append(json.loads(line))

    return data_id2data, data_id_set, mode


def run_eval(args,
        data_id2data, 
        data_id_set, 
        mode):
    # Evaluate the model for answers
    model, tokenizer = load_model(
        args.model_path, args.device
    )
    if "cuda" in args.device or args.device == "mps":
        model.to(args.device)
    # model = model.to(args.device)

    with open(args.answer_file, mode) as ans_file:
        for i, tid in enumerate(tqdm(data_id_set)):
            test = data_id2data[tid]
            qid = test['ID']
            input = test["input"]
            reference = test["ref"]
            input_ids = tokenizer([input]).input_ids
            output_ids = model.generate(
                torch.as_tensor(input_ids).to(args.device),
                # do_sample=True,
                # temperature=0.1,
                # num_beams=2,
                max_new_tokens=128,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                # force_words_ids=tokenizer(['A', 'B', 'C', 'D'], add_special_tokens=False, return_tensors="pt").input_ids.tolist()
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            # print("gen:",outputs)
            gen = {
                    "id": qid,
                    "input": input,
                    "reference": reference,
                    "generation": outputs,
                }
            ans_file.write(json.dumps(gen) + "\n")
    # Write output to file
    # with open(args.answer_file, "w") as ans_file:
    #     for line in gen:
    #         ans_file.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/media/public/models/huggingface/pythia-12b",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="wmt/test_wmt.json"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cuda:8",
        help="The device type",
    )
    parser.add_argument(
        "--answer_file",
        type=str,
        default="wmt/wmt_output_pythai.json"
    )
    args = parser.parse_args()

    data_id2data, data_id_set, mode = get_input(args.test_file, args.answer_file)
    run_eval(
        args,
        data_id2data, 
        data_id_set, 
        mode
    )