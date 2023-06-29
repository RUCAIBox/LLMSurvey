#!/bin/bash

python src/main.py \
-b config/benchmark/bbh10k.yaml \
-m config/model/chatgpt.yaml \
-l chatgpt.log \
model.api_key $1