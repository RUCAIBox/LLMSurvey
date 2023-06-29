import logging

from yacs.config import CfgNode

from .Model import Model

class DummyModel(Model):

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
        result = ipt
        return result
