import openai
import time
import json


openai.api_key = 'sk-'

def get_res_batch(context, instruction):
    prompt = instruction + context + " "
    while True:
        try:
            res = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=0.0
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
    
    # print(res["choices"][0]['text'].strip())
    return res["choices"][0]['text'].strip()


def get_dataset(file, instruction):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        for i in range(len(data)):
            context = data[i]["question"]
            reference = data[i]["references"]

            ans = get_res_batch(context, instruction)

            gen = {"input": context, "ground_truth": reference, "generation": ans}
            dump_jsonl(gen, "generation/text-davinci-003.json")

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
    file = "data/wikifact.json"
    instruction = "Complete the sentence with one or a few words. "
    get_dataset(file, instruction)
