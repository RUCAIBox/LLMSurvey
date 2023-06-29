#!/bin/bash

python src/main.py \
-b config/benchmark/bbh10k.yaml \
-m config/model/openai.yaml \
-l openai.log \
model.model_alias $1 \
model.api_key $2
