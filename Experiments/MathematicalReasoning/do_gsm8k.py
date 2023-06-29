#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import re
import json
import logging
import random
import os

import sys
from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm

import datasets
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from datasets import Dataset
from filelock import FileLock

import torch

import transformers
from transformers import (
    AutoConfig,
    LlamaConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    MBartTokenizerFast,
    LlamaTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    AutoModelForCausalLM,
)
from transformers.trainer_utils import IntervalStrategy
from transformers.trainer_callback import TrainerCallback
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

from chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm.tokenization_chatglm import ChatGLMTokenizer
from chatglm.configuration_chatglm import ChatGLMConfig

from falcon.modelling_RW import RWForCausalLM

from prompt_pattern import PROMPT, STOP_WORD

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    source_suffix: Optional[str] = field(
        default="", metadata={"help": "A suffix to add after every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    max_source_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    result_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )


@dataclass
class DataCollatorForMWP:
    def __init__(self, tokenizer, max_source_length, model_type='other'):
        self.pad_token_id = tokenizer.pad_token_id
        self.max_source_length = max_source_length
        self.model_type = model_type
    
    def __call__(self, examples):
        batch = {}

        batch['input_ids'] = []
        for e in examples:
            if (len(e['input_ids']) > self.max_source_length):
                e['input_ids'] = e['input_ids'][-self.max_source_length:]
            input_ids = torch.cat((
                torch.tensor([self.pad_token_id] * (self.max_source_length - len(e['input_ids'])), dtype=torch.int64),
                torch.tensor(e['input_ids'], dtype=torch.int64), 
            ), dim=0)
            input_ids = input_ids.unsqueeze(0)
            batch['input_ids'].append(input_ids)
        batch['input_ids'] = torch.cat(batch['input_ids'], dim=0)

        batch['attention_mask'] = torch.where(batch['input_ids'] == self.pad_token_id, 0, 1)
        if (self.model_type == 'chatglm'):
            batch['attention_mask'] = torch.tensor(batch['attention_mask'], dtype=torch.uint8)

        batch['labels'] = []
        for e in examples:
            labels = torch.cat((
                torch.tensor([self.pad_token_id] * (self.max_source_length - len(e['input_ids'])), dtype=torch.int64),
                torch.tensor(e['input_ids'], dtype=torch.int64), 
            ), dim=0)
            labels = labels.unsqueeze(0)
            batch['labels'].append(labels)
        batch['labels'] = torch.cat(batch['labels'], dim=0)

        return batch


class EvalEpochIntervalCallback(TrainerCallback):

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch)

        if (epoch % 5 == 0):
            control.should_save = True
        else:
            control.should_save = False

        if (args.logging_strategy == IntervalStrategy.EPOCH):
            control.should_log = True
        
        control.should_evaluate = True

        return control


def clean(content):
    pattern = '<<.+>>'
    result = re.findall(pattern, content)
    for t in result:
        content = content.replace(t, '')
    content = content.replace('\n', '. ')
    return content


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if ('llama' in model_args.model_name_or_path.lower()):
        pattern = PROMPT['VANILLA']
        stop = STOP_WORD['VANILLA']
    # elif ('alpaca' in model_args.model_name_or_path.lower()):
    #     pattern = PROMPT['ALPACA'].format('Solve math word problem.', '{}', '{}')
    #     stop = STOP_WORD['ALPACA']
    else:
        pattern = PROMPT['VANILLA']
        stop = STOP_WORD['VANILLA']

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    random.seed(training_args.seed)

    train_dataset = DatasetDict.from_json(data_args.dataset_path, field='train')
    ids = random.sample([i for i in range(len(train_dataset))], 3)
    demo = ''
    for idx in ids:
        data = train_dataset[idx]
        problem = data['question']
        solution = data['answer']
        answer = solution.split('####')[-1].strip()
        solution = clean(solution.split('####')[0].strip())
        demo = demo + pattern.format(problem, f'{solution} The answer is {answer}')
        # if ('vicuna' in model_args.model_name_or_path.lower()):
        #     demo = 'You are an helpful expert in mathematical problem. ' + 'USER' + ': ' + demo
        # elif ('pythia' in model_args.model_name_or_path.lower()):
        #     demo = 'You are an helpful expert in mathematical problem.' + ' <|prompter|> ' + demo

    logger.info(f'\n{demo}')

    test_dataset = DatasetDict.from_json(data_args.dataset_path, field='test')
    # test_dataset = Dataset.from_dict(test_dataset[:4])
    logger.info('--------------- Raw Dataset ---------------')
    logger.info(train_dataset)
    logger.info(test_dataset)

    # Load pretrained model and tokenizer
    if ('chatglm' in model_args.model_name_or_path):
        config = ChatGLMConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = ChatGLMTokenizer.from_pretrained(model_args.model_name_or_path)
    elif ('llama' not in model_args.model_name_or_path):
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            trust_remote_code=True,
        )
    else:
        config = LlamaConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    logger.info(config)
    logger.info(len(tokenizer))
    logger.info(tokenizer.pad_token)
    logger.info(tokenizer.bos_token)
    logger.info(tokenizer.eos_token)

    my_max_input_length = 0
    def process(data):
        nonlocal my_max_input_length
        prompt = demo + data['question']
        # if ('vicuna' in model_args.model_name_or_path.lower()):
        #     prompt = prompt  + ' ASSISTANT:'
        # elif ('pythia' in model_args.model_name_or_path.lower()):
        #     prompt = prompt + ' <|endoftext|> '  + ' <|assistant|> '
        inputs = tokenizer(prompt)
        labels = tokenizer(data['answer'])

        data['input_ids'] = inputs['input_ids']
        # print(len(data['input_ids']), inputs)
        my_max_input_length = max(my_max_input_length, len(data['input_ids']))
        print(my_max_input_length)
        # data['attention_mask'] = inputs['attention_mask']
        data['labels'] = labels['input_ids']
        
        return data
    
    logger.info('Max Input Length: {}'.format(my_max_input_length))

    # train_dataset = train_dataset.map(process)
    test_dataset = test_dataset.map(process)
    logger.info('--------------- Processed Dataset ---------------')
    # logger.info(train_dataset)
    logger.info(test_dataset)

    if (
        'gpt' in model_args.model_name_or_path or
        'llama' in model_args.model_name_or_path or 
        'alpaca' in model_args.model_name_or_path or
        'vicuna' in model_args.model_name_or_path or
        'pythia' in model_args.model_name_or_path
    ):
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.pad_token_id = tokenizer.bos_token_id
        config.pad_token_id = config.bos_token_id

    if ('falcon' in model_args.model_name_or_path):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        config.pad_token_id = config.eos_token_id
    
    if ('galactica' in model_args.model_name_or_path):
        tokenizer.pad_token_id = config.pad_token_id
        tokenizer.eos_token_id = config.eos_token_id
        tokenizer.bos_token_id = config.bos_token_id

    if (
        'bloom' in model_args.model_name_or_path or
        'gpt' in model_args.model_name_or_path or
        'llama' in model_args.model_name_or_path or
        'alpaca' in model_args.model_name_or_path or
        'vicuna' in model_args.model_name_or_path or
        'pythia' in model_args.model_name_or_path or
        'falcon' in model_args.model_name_or_path or
        'galactica' in model_args.model_name_or_path
    ):
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            trust_remote_code=True,
        )
    elif ('falcon' in model_args.model_name_or_path):
        model = RWForCausalLM.from_pretrained(
            model_args.model_name_or_path,
        )
    else:
        model = ChatGLMForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
        )

    if ('chatglm' in model_args.model_name_or_path):
        model = model.half()
    logger.info(model)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    # Data collator
    if ('chatglm' in model_args.model_name_or_path):
        data_collator = DataCollatorForMWP(tokenizer, data_args.max_source_length, 'chatglm')
    else:
        data_collator = DataCollatorForMWP(tokenizer, data_args.max_source_length)

    best_em = 0.0
    def compute_metrics(eval_preds):
        nonlocal best_em

        preds, labels = eval_preds

        print(len(tokenizer), model.get_input_embeddings().weight.shape[0])

        print(type(preds))
        preds = preds.tolist()

        for i in range(len(preds)):
            preds[i] = preds[i][data_args.max_source_length:]
            for j in range(len(preds[i])):
                assert (preds[i][j] < len(tokenizer))
                assert (preds[i][j] >= 0 or preds[i][j] == -100)
                if (preds[i][j] == -100):
                    preds[i][j] = tokenizer.pad_token_id
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = []
        for data in test_dataset:
            decoded_labels.append(data['answer'])

        fout = open(data_args.result_path, 'w')

        num_correct = 0
        total_problem = 0
        for pred, label, data in zip(decoded_preds, decoded_labels, test_dataset):
            tmp_data = {
                'question': data['question'],
                'solution': data['answer'],
            }

            if (stop in pred):
                pred = pred.split(stop)[0].strip()
            tmp_data['prediction'] = pred

            if ('The answer is' in pred): 
                pred_ans = pred.split('The answer is')[-1].strip()
                if (len(pred_ans) == 0):
                    pred_ans = ' '
                if (pred_ans[-1] == '.'):
                    pred_ans = pred_ans[:-1]
            else: pred_ans = ' '
            
            label_ans = label.split('####')[-1].strip()

            tmp_data['pred_ans'] = pred_ans
            tmp_data['real_ans'] = label_ans

            tmp_data['score'] = False
            if (pred_ans == label_ans):
                num_correct = num_correct + 1
                tmp_data['score'] = True
            total_problem = total_problem + 1

            fout.write(json.dumps(tmp_data, indent=4) + '\n')

        fout.close()

        result = round(num_correct / total_problem * 100, 2)
        best_em = max(best_em, result)
        logger.info(f'Best Exactly Match: {best_em}')
        return {'EM': result}

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[]
    )

    # Predict
    predict_results = trainer.predict(test_dataset, metric_key_prefix="predict")
    logger.info(predict_results)


if __name__ == "__main__":
    main()