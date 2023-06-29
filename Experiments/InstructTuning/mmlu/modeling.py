import getpass
from pathlib import Path
from typing import Optional

import openai
import tiktoken
import torch
import torch.nn as nn
import transformers
from fire import Fire
from peft import PeftModel
from pydantic import BaseModel
from torchvision.datasets.utils import download_url
from transformers import AutoTokenizer
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModel,
    LlamaConfig,
)
import quant


class EvalModel(BaseModel, arbitrary_types_allowed=True):
    model_path: str
    max_input_length: int = 512
    max_output_length: int = 512

    def run(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def count_text_length(self, text: str) -> int:
        raise NotImplementedError

    def check_valid_length(self, text: str) -> bool:
        return self.count_text_length(text) <= self.max_input_length


class OpenAIModel(EvalModel):
    model_path: Optional[str]
    api_key: Optional[str]
    use_azure: bool = False
    tokenizer: Optional[tiktoken.Encoding]
    api_endpoint: str = "https://research.openai.azure.com/"
    api_version: str = "2023-03-15-preview"

    def load(self):
        if self.tokenizer is None:
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model_path)
            except Exception as e:
                print(e)
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # chatgpt/gpt-4

        if self.use_azure:
            openai.api_type = "azure"
            openai.api_base = self.api_endpoint
            openai.api_version = self.api_version
            if self.model_path is None:
                self.model_path = getpass.getpass("Model or deployment name: ")

        if self.api_key is None:
            self.api_key = getpass.getpass("API Key: ")
        openai.api_key = self.api_key

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        try:
            response = openai.ChatCompletion.create(
                engine=self.model_path,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            return str(e)

    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer.encode(text))


class SeqToSeqModel(EvalModel):
    model_path: str
    model: Optional[PreTrainedModel]
    tokenizer: Optional[PreTrainedTokenizer]
    lora_path: str = ""
    device: str = "cuda"
    load_8bit: bool = False
    quant_path: str = ""

    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, **args)
            if self.lora_path:
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model.eval()
            if not self.load_8bit:
                self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_output_length,
            **kwargs,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer(text).input_ids)


class CausalModel(SeqToSeqModel):
    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True, **args
            )
            self.model.eval()
            if not self.load_8bit:
                self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)


class LlamaModel(SeqToSeqModel):
    use_template: bool = False
    """
    Not officially supported by AutoModelForCausalLM, so we need the specific class
    Optionally, we can use the prompt template from: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
    However, initial MMLU experiments indicate that the template is not useful for few-shot settings
    """

    def load(self):
        if self.tokenizer is None:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        if self.model is None:
            if self.quant_path:
                self.model = load_quant(self.model_path, self.quant_path, wbits=2, groupsize=128)
                self.model.to(self.device)
                self.model.eval()
                
            else:
                args = {}
                if self.load_8bit:
                    args.update(device_map="auto", load_in_8bit=True)
                self.model = LlamaForCausalLM.from_pretrained(self.model_path, **args)
                if self.lora_path:
                    self.model = PeftModel.from_pretrained(self.model, self.lora_path)
                self.model.eval()
                if not self.load_8bit:
                    self.model.to(self.device)
            

    def run(self, prompt: str, **kwargs) -> str:
        if self.use_template:
            template = (
                "The following is a conversation between a human and an AI assistant. "
                "The AI assistant gives helpful, detailed, and polite answers to the user's questions.\n"
                "[|Human|]: {instruction}\n\n[|AI|]:"
            )
            text = template.format_map(dict(instruction=prompt))
        else:
            text = prompt

        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        if "65b" in self.model_path.lower():
            self.max_input_length = 1024
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length,
            ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)


def find_layers(module, layers=(nn.Conv2d, nn.Linear), name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def noop(*args, **kwargs):
    assert args is not None
    assert kwargs is not None


def load_quant(
    model,
    checkpoint,
    wbits,
    groupsize=-1,
    fused_mlp=True,
    warmup_autotune=True,
):
    config = LlamaConfig.from_pretrained(model)
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop
    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()

    layers = find_layers(model)
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]

    quant.make_quant_linear(model, layers, wbits, groupsize)
    del layers

    print("Loading model ...")
    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load

        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)
    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)

    model.seqlen = 2048
    print("Done.")
    return model


class GPTQModel(LlamaModel):
    quantized_path: str
    model: Optional[LlamaForCausalLM]
    tokenizer: Optional[LlamaTokenizer]
    num_bits: int = 4
    group_size: int = 128

    def load(self):
        # https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/05781593c818d4dc8adc2d32c975e83d17d2b9a8/llama_inference.py
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not Path(self.quantized_path).exists():
            url = f"https://huggingface.co/{self.model_path}/resolve/main/{self.quantized_path}"
            download_url(url, root=".")

        if self.model is None:
            self.model = load_quant(
                model=self.model_path,
                checkpoint=self.quantized_path,
                wbits=self.num_bits,
                groupsize=self.group_size,
            )
            self.model.to(self.device)

        if self.tokenizer is None:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
            self.test_max_length()

    def test_max_length(self):
        # Detect any OOMs at the beginning
        text = " ".join(["test sentence for max length"] * 1000)
        self.run(text)


class ChatGLMModel(SeqToSeqModel):
    def load(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        if self.model is None:
            self.model = AutoModel.from_pretrained(
                self.model_path, trust_remote_code=True
            ).half()  # FP16 is required for ChatGLM
            self.model.eval()
            self.model.to(self.device)

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        response, history = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            **kwargs,
        )
        return response


def select_model(model_name: str, **kwargs) -> EvalModel:
    model_map = dict(
        seq_to_seq=SeqToSeqModel,
        causal=CausalModel,
        llama=LlamaModel,
        chatglm=ChatGLMModel,
        openai=OpenAIModel,
        gptq=GPTQModel,
    )
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"{model_name}. Choose from {list(model_map.keys())}")
    return model_class(**kwargs)


def test_model(
    prompt: str = "Write an email about an alpaca that likes flan.",
    model_name: str = "seq_to_seq",
    model_path: str = "google/flan-t5-base",
    **kwargs,
):
    model = select_model(model_name, model_path=model_path, **kwargs)
    print(locals())
    print(model.run(prompt))


if __name__ == "__main__":
    Fire()
