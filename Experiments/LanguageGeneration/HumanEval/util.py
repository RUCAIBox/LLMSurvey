import signal
import time
from typing import Dict

import openai
openai.api_key = 'sk-'

# TODO Codex request if we need it.
def create_chatgpt_config(
    message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    config = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": batch_size,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ],
    }
    return config

def create_davinci_config(
    # message: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    config = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": batch_size,
        # "messages": [
        #     {"role": "system", "content": system_message},
        #     {"role": "user", "content": message},
        # ],
        "prompt": prompt,
    }
    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def request_chatgpt_engine(config) -> Dict:
    ret = None
    while ret is None:
        try:
            ret = openai.ChatCompletion.create(**config)
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)
    
    return ret

def request_gpt3_engine(config) -> Dict:
    ret = None
    while ret is None:
        try:
            ret = openai.Completion.create(**config)
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)
    
    return ret
