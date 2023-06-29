import logging

import openai
from yacs.config import CfgNode

from .Model import Model

class OpenAIModel(Model):

    def __init__(self, config: CfgNode, logger: logging.Logger):
        super().__init__(config, logger)
        self.logger = logger
        self.config = config
        self.model_alias = self.config.model.model_alias
        assert self.model_alias is not None, "please specify the model_alias in the config file"
        assert config.model.api_key is not None, "please specify the api_key in the config file"

    def generate_text(self, ipt: str, *args, **kwargs) -> str:
        try:
            openai.api_key = self.config.model.api_key
            response = openai.Completion.create(
                model=self.config.model.model_alias,
                prompt=ipt,
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens,
                n=self.config.model.n,
                logprobs=self.config.model.logprobs
            )
            #response = self.wait_for_complete(response["id"])
            result=response['choices'][0]['text']
            return result
        except openai.error.OpenAIError as e:
            self.logger.error(f"OpenAI error:  {e}")
            return ""
