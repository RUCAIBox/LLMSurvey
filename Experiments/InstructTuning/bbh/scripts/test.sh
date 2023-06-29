#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python src/main.py \
-b config/benchmark/bbh10k.yaml \
-m config/model/llama.yaml \
-l new_logs/difficulty.log \
model.model_alias llama-model \
model.url http://localhost:5000
 