import argparse
import random
import sys
import json
import openai
import anthropic
import multiprocessing as mp
import time

import requests

from utils import encode_question, change_api


def get_response(get_response_input, api_key):
    question, question_id, api_name, model = get_response_input
    question = encode_question(question, api_name)

    while True:
        try:
            if "gpt" in model:
                openai.api_key = api_key
                responses = openai.ChatCompletion.create(
                    model=model,
                    messages=question,
                    n=1,
                    temperature=0,
                )
                response = responses['choices'][0]['message']['content']
            elif "claude" in model:
                client = anthropic.Client(api_key)
                responses = client.completion(
                    prompt=f"{anthropic.HUMAN_PROMPT} {question[0]['content']}{question[1]['content']}{anthropic.AI_PROMPT}",
                    stop_sequences=[anthropic.HUMAN_PROMPT],
                    model=model,
                    max_tokens_to_sample=1024,
                )
                response = responses["completion"].strip()
            elif "davinci" in model:
                openai.api_key = api_key
                prompt = "###USER: " + f"{question[0]['content']}{question[1]['content']}" + "###ASSISTANT: "
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=1024,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                response = response["choices"][0]["text"]
            else:
                response = None
                print("Error: Model is not supported.")
            break
        except openai.error.RateLimitError as e:
            if str(e) == "You exceeded your current quota, please check your plan and billing details.":
                change_api()
                time.sleep(10)
            else:
                print('\nopenai.error.RateLimitError\nRetrying...')
                time.sleep(10)
        except openai.error.ServiceUnavailableError:
            print('\nopenai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(10)
        except openai.error.Timeout:
            print('\nopenai.error.Timeout\nRetrying...')
            time.sleep(10)
        except openai.error.APIError:
            print('\nopenai.error.APIError\nRetrying...')
            time.sleep(10)
        except openai.error.APIConnectionError:
            print('\nopenai.error.APIConnectionError\nRetrying...')
            time.sleep(10)
        except anthropic.ApiException:
            print('\nanthropic.ApiException\nRetrying...')
            time.sleep(10)
        except requests.exceptions.ProxyError:
            print('\nrequests.exceptions.ProxyError\nRetrying...')
            time.sleep(10)
        except Exception as e:
            print(e)
            return None

    print("=>", question_id)
    return {"question_id": question_id, 'text': response, "answer_id": "None", "model_id": model, "metadata": {}}

def process_entry(entry, api_key):
    question, question_id, api_name, model = entry
    result = get_response((question, question_id, api_name, model), api_key)
    return result


def write_result_to_file(result, output_file):
    global file_write_lock
    with file_write_lock:
        with open(output_file, "a") as outfile:
            json.dump(result, outfile)
            outfile.write("\n")


def callback_with_lock(result, output_file):
    global file_write_lock
    write_result_to_file(result, output_file, file_write_lock)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="which model you want to use for eval, only support ['gpt*', 'claude*'] now")
    parser.add_argument("--api_key", type=str, default="", help="the api key provided for calling")
    parser.add_argument("--output_file", type=str, default="./eval-data/llm-responses/gpt-3.5-turbo_torchhub_0_shot.jsonl", help="the output file this script writes to")
    parser.add_argument("--question_file", type=str, default="eval-data/questions/torchhub/questions_torchhub_0_shot.jsonl", help="path to the questions data file")
    parser.add_argument("--api_name", type=str, default="torchhub", help="this will be the api dataset name you are testing, only support ['torchhub', 'tensorhub', 'huggingface'] now")
    args = parser.parse_args()

    start_time = time.time()
    # Read the question file
    questions = []
    question_ids = []
    with open(args.question_file, 'r') as f:
        for idx, line in enumerate(f):
            questions.append(json.loads(line)["text"])
            question_ids.append(json.loads(line)["question_id"])

    file_write_lock = mp.Lock()
    with mp.Pool(1) as pool:
        results = []
        for idx, (question, question_id) in enumerate(zip(questions, question_ids)):
            result = pool.apply_async(
                process_entry,
                args=((question, question_id, args.api_name, args.model), args.api_key),
                callback=lambda result: write_result_to_file(result, args.output_file),
            )
            results.append(result)
        pool.close()
        pool.join()

    end_time = time.time()
    print("Total time used: ", end_time - start_time)