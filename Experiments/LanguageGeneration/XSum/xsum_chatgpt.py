import openai
import time
import json
import tiktoken

openai.api_key = 'sk-'

def num_tokens_from_message(message, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(message))
    return num_tokens

def get_res_batch(context, instruction):
    content = context + instruction + " "
    
    # truncation
    if num_tokens_from_message(content) > 4000-128:
        truncation_length = 4000-128
        while(num_tokens_from_message(content)>truncation_length):
            content = " ".join(content.split()[:-1])
    message = [
        {"role": "user", "content": content} 
    ]
    while True:
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=0.0,
                max_tokens=128,
            )
            break
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
    
    
    # print(res['choices'][0]['message']['content'])
    return res['choices'][0]['message']['content']


def get_dataset(file, instruction):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        for i in range(len(data)):
            context = data[i]["dialogue"]
            reference = data[i]["summary"]

            ans = get_res_batch(context, instruction)

            gen = {"input": context, "ground_truth": reference, "generation": ans}
            dump_jsonl(gen, "generation/gpt-3.5-turbo.json")

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, 'a+', encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')
    # print('Wrote {} records to {}'.format(len(data), output_path))


if __name__ == '__main__':
    file = "data/xsum.json"
    instruction = "Try your best to summarize the main content of the given document. And generate a short summary in 1 sentence for it."
    get_dataset(file, instruction)
