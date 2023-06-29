#!/bin/bash

python src/main.py \
-b config/benchmark/bbh10k.yaml \
-m config/model/dummy.yaml \
-l dummy.log \
model.model_alias dummy-model \
model.url http://localhost:5000
 