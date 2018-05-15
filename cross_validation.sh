#!/usr/bin/env bash

DIR_DATA=./data
DIR_WORK=./work
FILE_DATASET=dataset.json
FILE_FEATURES=features.json
FILE_REDIRECTS=redirects.json

N_FEATURE=10000
VECTOR_SIZE=200
HIDDEN_SIZE=200
GPU_ID=0

python scripts/cross_validation.py --dataset $DIR_DATA/$FILE_DATASET --features $DIR_WORK/$FILE_FEATURES --redirects $DIR_WORK/$FILE_REDIRECTS --out_dir $DIR_WORK --gpu $GPU_ID --n_feature $N_FEATURE --embed_size $VECTOR_SIZE --hidden_size $HIDDEN_SIZE
