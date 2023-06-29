# Mathematical Reasoning

## Datasets

We provide the processed dataset in folder `dataset/` and few-shot exemplars in `demo/`.

## Models

You have to download the checkpoints in hugging face version, and save them in folder `model/` with the model name as the sub-folder name.

## Environment

Here are the main environmental configurations
```
python==3.9.13
torch==1.12.1
transformers==4.29.1
datasets==2.10.1
sympy==1.11.1
openai
```
Specially, for running falcon on our code, you need pytorch 2.x version instead of pytorch 1.x.
```
torch==2.0.1
```

## Evaluation

You can running the scripts in folder `scripts/` to evaluate the LLMs and reproduce our experiment:

```
bash scripts/run_eval_color.sh
bash scripts/run_eval_penguins.sh
bash scripts/run_eval_gsm8k.sh
bash scripts/run_eval_math.sh
```

To better evaluate the results from prediction of LLMs with the golden labels, we use `evaluate.py` to clean the results from LLMs and evaluate the accuracy. You can find the details in the corresponding file.