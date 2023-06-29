import logging

import openai
from yacs.config import CfgNode

from .Model import Model

class ChatGPTModel(Model):

    def __init__(self, config: CfgNode, logger: logging.Logger):
        super().__init__(config, logger)
        self.logger = logger
        self.config = config
        self.model_alias = "gpt-3.5-turbo"
        assert config.model.api_key is not None, "please specify the api_key in the config file"

    def generate_text(self, ipt: str, *args, **kwargs) -> str:
        try:
            openai.api_key = self.config.model.api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": ipt}
                ],
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens,
                n=self.config.model.n
            )
            #response = self.wait_for_complete(response["id"])
            result=response["choices"][0]["message"]["text"]
            return result
        except openai.error.OpenAIError as e:
            self.logger.error(f"OpenAI error:  {e}")
            return ""
