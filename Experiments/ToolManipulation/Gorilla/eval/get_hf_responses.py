import json
import argparse
import os
import warnings

from tqdm import tqdm
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from utils import encode_question


# Load Model from HF
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
        )
    except ValueError:
        model = AutoModel.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **kwargs,
        )
    except Exception as e:
        print(e)
        return None

    return model, tokenizer


def get_questions(question_file):
    # Load questions file
    question_jsons = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            question_jsons.append(line)

    return question_jsons


def run_eval(args, question_jsons):
    # Evaluate the model for answers
    model, tokenizer = load_model(
        args.model_path, args.device
    )
    if "cuda" in args.device or args.device == "mps":
        model.to(args.device)

    ans_jsons = []
    for i, line in enumerate(tqdm(question_jsons)):
        ques_json = json.loads(line)
        idx = ques_json["question_id"]
        question = ques_json["text"]
        question = encode_question(question,args.api_name)
        prompt = "###USER: " + f"{question[0]['content']}{question[1]['content']}" + "###ASSISTANT: "
        inputs = tokenizer([prompt])
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        # print(len(input_ids),len(attention_mask))
        output_ids = model.generate(
            input_ids=torch.as_tensor(input_ids).to(args.device),
            attention_mask=torch.as_tensor(attention_mask).to(args.device),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
            early_stopping=True,
            # eos_token_id=tokenizer.eos_token_id,
        )
        output_ids = output_ids[0][len(input_ids[0]):]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        ans_jsons.append(
            {
                "text": outputs,
                "question_id": idx,
                "questions": prompt,
            }
        )

    # Write output to file
    with open(args.answer_file, "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")

    return ans_jsons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--question_file",
        type=str,
        default="eval-data/questions/torchhub/questions_torchhub_0_shot.jsonl"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cuda:0",
        help="The device type",
    )
    parser.add_argument(
        "--answer_file",
        type=str,
        default="eval-data/llm-responses/torchhub_0_shot.jsonl"
    )
    parser.add_argument(
        "--api_name",
        type=str,
        default="torchhub",
        help="this will be the api dataset name you are testing, only support ['torchhub', 'tensorhub', 'huggingface'] now"
    )
    args = parser.parse_args()

    questions_json = get_questions(args.question_file)
    run_eval(
        args,
        questions_json
    )