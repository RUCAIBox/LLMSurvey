import time
import json
import slack
import argparse
import warnings
from flask import Flask
from flask_slack import Slack

app = Flask(__name__)
slack_app = Slack(app)


def get_history():
    
   history = client.conversations_history(channel=channel_id)
   text = history['messages'][0]['text']
   return text

def annotate(message):
    # send to defined channel
    client.chat_postMessage(channel=channel_id, text=message, as_user=True)

    # get response
    text = get_history()
    temp = ''
    while True:
        temp = get_history()
        if temp != text and 'Typing' not in temp:
            break
        else:
            time.sleep(1)
    response = temp
    return response

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, 'a+', encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')

if __name__ == '__main__':
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--userOAuthToken',default="xoxp-")
    parser.add_argument('--channel_id', default="")
    parser.add_argument('--dataset', default="lambada")
    
    args = parser.parse_args()
    
    channel_id = args.channel_id
    userOAuthToken = args.userOAuthToken
    client = slack.WebClient(token=userOAuthToken)
    
    dataset = args.dataset
    save_dir = f'generation/Claude.json'
    
    
    if dataset == 'lambada':
        ins = "I want you act as a sentence completer. Given a passage, your objective is to add one word(no punctuation) at the end of the last sentence to complete the sentence and ensure its semantic integrity."
        eval_data_path = f"data/lambada_test.json"
        with open(eval_data_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            cnt=0
            for i, line in enumerate(lines):
                each_data = json.loads(line)
                input = ins + " " + each_data['context']

                ans = ''
                while len(ans) == 0:
                    try:
                        ans = annotate(input)
                    except:
                        continue

                ans = ans.replace("\n", "").strip()
                if ans.split(" ")[-1] == " ":
                    ans = ans.split(" ")[-2]
                else:
                    ans = ans.split(" ")[-1]
                ans = ans.replace(".", " ").replace(",", " ").replace("?", " ").replace("!", " ").replace(":", " ").replace("\"","")
                ans = ans.strip()

                if ans.lower() == each_data["target_word"].lower():
                    cnt = cnt + 1

                dump_jsonl({"input": input, "reference": each_data['target_word'], "generation": ans}, save_dir)
            print("Claude Accuracy:", cnt/5153)
            

              