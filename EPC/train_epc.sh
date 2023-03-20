#!/bin/sh
exp_name="epc_noise0"
noise=0.0


CUDA_VISIBLE_DEVICES="" python -m maddpg_o.experiments.train_epc1 \
    --initial-population=3 \
    --num-selection=2 \
    --num-stages=3 \
    --stage-num-episodes 100000 50000 50000 \
    --num-good=3 \
    --num-food=3 \
    --good-policy=att-maddpg \
    --save-dir="./result_new/$exp_name" \
    --save-rate=1000 \
    --train-rate=1000 \
    --n-cpu-per-agent=8 \
    --noise-std=$noise \
    2>&1 | tee exp_logs_new/${exp_name}.log