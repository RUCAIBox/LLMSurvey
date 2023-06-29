import logging

from yacs.config import CfgNode

class Model:

    def __init__(self, config: CfgNode, logger: logging.Logger):
        self.model_alias = "base-model"

    def generate_text(self, ipt: str, *args, **kwargs) -> str:
        pass
