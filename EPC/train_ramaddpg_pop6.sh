#!/bin/sh
exp_name="ramaddpg_mutation_noise3_pop6"
noise=3

CUDA_VISIBLE_DEVICES="" python -m maddpg_o.experiments.train_epc1 \
    --initial-population=6 \
    --num-selection=3 \
    --num-stages=3 \
    --stage-num-episodes 100000 50000 50000\
    --num-good=3 \
    --num-food=3 \
    --good-policy=r-att-maddpg \
    --save-dir="./result/$exp_name" \
    --save-rate=1000 \
    --train-rate=1000 \
    --n-cpu-per-agent=60 \
    --noise-std=$noise \
    2>&1 | tee ${exp_name}.log