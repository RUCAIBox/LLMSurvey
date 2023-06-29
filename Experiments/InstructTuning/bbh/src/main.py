import os
import logging
import argparse
from yacs.config import CfgNode
import IPython
import model
from BBH10KBenchmark import BBH3kBenchmark

model_names = sorted(
    name
    for name in model.__dict__
    if name.islower() and not name.startswith("__") and callable(model.__dict__[name])
)

def set_logger(log_file, name="default"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.FileHandler(log_file, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger

def add_variable_to_config(cfg: CfgNode, name: str, value) -> CfgNode:
    cfg.defrost()
    cfg[name] = value
    cfg.freeze()
    return cfg

def load_cfg(cfg_file: str, new_allowed: bool=True) -> CfgNode:
    with open(cfg_file, "r") as fi:
        cfg = CfgNode.load_cfg(fi)
    cfg.set_new_allowed(new_allowed)
    return cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--benchmark-cfg", 
        type=str, 
        required=True, 
        help="Path to task config file"
    )
    parser.add_argument(
        "-m",
        "--model-cfg",
        type=str,
        required=True,
        help="Path to model config file"
    )
    parser.add_argument(
        "-l",
        "--log-file", 
        type=str, 
        default="log.log", 
        help="Path to log file"
    )
    parser.add_argument(
        "-p", 
        "--model-path",
        default=None, 
        help="path to fine-tuned model"
    )
    parser.add_argument(
        "opts", 
        default=None, 
        nargs=argparse.REMAINDER, 
        help="Modify config options from command line"
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    logger = set_logger(args.log_file, str(os.getpid()))
    logger.info(f"os.getpid()={os.getpid()}")
    config = CfgNode(new_allowed=True)
    config = add_variable_to_config(
        config, 
        "benchmark", 
        load_cfg(args.benchmark_cfg)
    )
    config = add_variable_to_config(
        config,
        "model",
        load_cfg(args.model_cfg)
    )
    config.merge_from_list(args.opts)
    logger.info(f"\n{config}")
    benchmark = BBH3kBenchmark(config, logger)
    model_cls = model.__dict__[config.model.model_cls]
    config.defrost()
    config.model.model_path = args.model_path
    config.freeze()
    print(f'loading model from {config.model.model_path}')
    benchmark.evaluate_model(model_cls(config, logger))


if __name__ == "__main__":
    main()
