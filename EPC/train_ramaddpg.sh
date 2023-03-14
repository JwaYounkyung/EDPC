#!/bin/sh
exp_name="ramaddpg_noise0"
noise=0

CUDA_VISIBLE_DEVICES="" python -m maddpg_o.experiments.train_epc1 \
    --initial-population=3 \
    --num-selection=2 \
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