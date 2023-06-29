import os
import sys

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import json
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

template = (
        "The following is a conversation between a human and an AI assistant. "
        "The AI assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        "[|Human|]: {instruction}\n\n[|AI|]:"
    )

def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    share_gradio: bool = False,
    output_file: str= "generates/default.json",
    input_file: str='alpaca_data_cleaned.json',
    model_name: str="none"
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    model.eval()

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        **kwargs,
    ):
        prompt = instruction['instruction']
        if instruction['input']:
            prompt += " Input: {instruction['input']}"
        prompt = template.format_map(dict(instruction=prompt))
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output.split("\n\n[|AI|]:")[1].strip().strip('</s>')


    instructions = json.load(open(input_file, 'r'))
    results = []
    for instruction in tqdm(instructions):
        instruction['generator'] = model_name
        instruction['output'] = evaluate(instruction)
        print("Instruction:", instruction)
        print("Response:", instruction['output'])

    json.dump(instructions, open(output_file, 'w'))



if __name__ == "__main__":
    fire.Fire(main)
