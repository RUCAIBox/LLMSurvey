import time
import json
import logging
import requests
from typing import Union, List, Optional

from yacs.config import CfgNode

from .Model import Model

class LocalModel(Model):

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

    def generate_text(self, ipt: str, *args, **kwargs) -> str:
        max_length = self.config.model.max_length
        try:
            # 发起completion请求
            data = {"input": [ipt], "max_length": max_length}
            response = requests.post(
                self.LOCAL_URL, headers=self.headers, data={"json":  json.dumps(data)}
            )
            result = response.json()["output"][0]
            return result
        except Exception as e:
            self.logger.exception(e)
            return ""
