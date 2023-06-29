#!/bin/bash

python src/main.py \
-b config/benchmark/bbh10k.yaml \
-m config/model/local.yaml \
-l local.log \
model.model_alias $1 \
model.url $2
