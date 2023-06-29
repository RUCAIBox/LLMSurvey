import argparse
from torchmetrics.text.rouge import ROUGEScore
import numpy as np
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate File")

    parser.add_argument("--test_file", help="test file")
    args = parser.parse_args()

    rougeScore = ROUGEScore()
    rougel = []
    with open(args.test_file) as f:
        for line in f:
            recover=json.loads(line)['generation']
            reference=json.loads(line)['ground_truth']
            rougel.append(rougeScore(recover, reference)['rougeL_fmeasure'].tolist())
    print('{} avg ROUGE-L score'.format(args.test_file.split("."[0])), np.mean(rougel))