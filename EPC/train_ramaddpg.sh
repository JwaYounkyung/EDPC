#!/bin/sh
exp_name="ramaddpg_mixexp0.25_noise3"
noise=3
mutation_rate=0.25
mode="proportional"

CUDA_VISIBLE_DEVICES="" python -m maddpg_o.experiments.train_epc1 \
    --initial-population=3 \
    --num-selection=2 \
    --num-stages=3 \
    --stage-num-episodes 100000 50000 50000\
    --mutation-rate=$mutation_rate \
    --num-good=3 \
    --num-food=3 \
    --good-policy=r-att-maddpg \
    --save-dir="./result/$exp_name" \
    --save-rate=1000 \
    --train-rate=1000 \
    --n-cpu-per-agent=60 \
    --noise-std=$noise \
    --roulette-mode=$mode \
    2>&1 | tee ${exp_name}.log