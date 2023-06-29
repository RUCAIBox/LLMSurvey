import argparse
import json
import re
import string


def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))

def exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return 1 if gold.strip() == pred.strip() else 0


def quasi_exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return 1 if normalize_text(gold) == normalize_text(pred) else 0


def prefix_exact_match(gold: str, pred: str) -> float:
    """
    The `prefix_exact_match` metric is particularly useful in the zero-shot setting, where the model is
    not given examples of the expected outputs and tends to output more tokens than it should.

    For example, for this zero-shot prompt from BoolQ,

    Passage: Elmendorf Air Force Base (IATA: EDF, ICAO: PAED, FAA LID: EDF) is a United States military facility
    in Anchorage, the largest city in Alaska. Originally known as Elmendorf Field, it became Elmendorf Air Force
    Base after World War II, and in 2010 it merged with nearby Fort Richardson to form Joint Base Elmendorf-Richardson.
    Question: Is there an air force base in anchorage alaska?
    Answer:

    the model could output up to `max_tokens` number of tokens "Yes, Elmendorf" instead of just "Yes".
    """
    if not pred:
        return 0

    return 1 if pred.strip().startswith(gold.strip()) else 0


def quasi_prefix_exact_match(gold: str, pred: str) -> float:
    """
    Same thing as `prefix_exact_match` but we normalize the text before checking if the prefix match.
    """
    if not pred:
        return 0

    return 1 if normalize_text(pred).startswith(normalize_text(gold)) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate File")

    parser.add_argument("--test_file", help="test file")
    args = parser.parse_args()
    test_file = args.test_file
    em_score = []
    with open(test_file,'r', encoding="utf-8") as f:
        for line in f:
            test = json.loads(line)
            input = test["input"]
            references = test["ground_truth"]
            generation = test["generation"].split("\n")[0]
            score = 0
            for reference in references:
                if reference.strip() in generation.strip():
                    score = 1 + score
            if score > 0:
                em_score.append(1)
            else:
                em_score.append(0)
    print("{} EM: ".format(test_file.split(".")[0]), sum(em_score)/len(em_score))
        

