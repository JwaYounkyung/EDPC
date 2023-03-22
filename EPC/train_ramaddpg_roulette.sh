#!/bin/sh
exp_name="ra_maddpg_roulette_testtest"
noise=3.0
# mode="proportional"
mode="ranking"

CUDA_VISIBLE_DEVICES="" python -m maddpg_o.experiments.train_epc1_roulette \
    --initial-population=3 \
    --num-selection=2 \
    --num-stages=3 \
    --stage-num-episodes 100000 50000 50000\
    --num-good=3 \
    --num-food=3 \
    --good-policy=r-att-maddpg \
    --save-dir="./result_new/$exp_name" \
    --save-rate=1000 \
    --train-rate=1000 \
    --n-cpu-per-agent=60 \
    --noise-std=$noise \
    --roulette-mode=$mode \
    2>&1 | tee exp_logs_new/${exp_name}.log