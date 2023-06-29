import argparse
import json
from sacrebleu.metrics import BLEU

length = [1000,1000,1000,1000,1000,1000,1000,1000,420,1000,1000,1000,1000,1000,420,1000,1000,1000,1000,1000,1000]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate File")

    parser.add_argument("--test_file", help="test file")
    args = parser.parse_args()
    
    reference = []
    generation = []
    results = []
    bleu = BLEU()
    cnt = 0
    i = 0
    with open(args.test_file, 'r', encoding="utf-8") as f:
            for line in f:
                cnt = cnt + 1      
                line = json.loads(line)
                reference.append([line["ground_truth"]])
                generation.append(line["generation"].replace("\n",""))
                if cnt >= length[i]:
                    if i == 7:
                        bleu = BLEU(tokenize='ja-mecab')
                    if i == 11:
                        bleu = BLEU(tokenize='zh')
                    else:
                        bleu = BLEU()

                    results.append(bleu.corpus_score(generation, reference))
                    cnt = 0 
                    reference = []
                    generation = []
                    i = i + 1
    sum = 0
    for i in range(len(results)):
        print(results[i].score)
        sum = sum + results[i].score
    print('{} BLEU-4 score'.format(args.test_file.split("."[0])), sum/len(results))