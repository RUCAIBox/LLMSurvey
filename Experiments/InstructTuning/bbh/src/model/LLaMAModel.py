import logging

from yacs.config import CfgNode

from .Model import Model

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
import IPython

class LLaMAModel(Model):

    def __init__(self, config: CfgNode, logger: logging.Logger) -> None:
        super().__init__(config, logger)
        self.logger = logger
        self.config = config
        self.LOCAL_URL = self.config.model.url
        assert self.LOCAL_URL is not None, "please specify the url in the config file"
        self.model_alias = self.config.model.model_alias
        assert self.model_alias is not None, "please specify the model_alias in the config file"
        self.headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        self.tokenizer = LlamaTokenizer.from_pretrained(self.config.model.model_path)
        self.model = LlamaForCausalLM.from_pretrained(self.config.model.model_path)
        self.max_output_length = 128
        self.device = "cuda"
        self.model.to(self.device)
        
        
    def generate_text(self, ipt: str, *args, **kwargs) -> str:
        template = (
                "The following is a conversation between a human and an AI assistant. "
                "The AI assistant gives helpful, detailed, and polite answers to the user's questions.\n"
                "[|Human|]: {instruction}\n\n[|AI|]:"
            )
        text = template.format_map(dict(instruction=ipt))
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)
