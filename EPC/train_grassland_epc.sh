#!/bin/sh

exp_name="grassland_epc_noise0"
noise=0.0


CUDA_VISIBLE_DEVICES="" python -m maddpg_o.experiments.train_epc1 \
    --scenario=grassland \
    --sight=100.0 \
    --initial-population=9 \
    --num-selection=2 \
    --num-stages=3 \
    --test-num-episodes=2000 \
    --stage-num-episodes 100000 50000 50000 \
    --num-good=3 \
    --num-adversaries=2 \
    --num-units=32 \
    --checkpoint-rate=0 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --save-dir="./result_new/$exp_name" \
    --save-rate=1000 \
    --train-rate=1000 \
    --n-cpu-per-agent=40 \
    --timeout=0.03 \
    --noise-std=$noise \
    2>&1 | tee exp_logs_new/${exp_name}.log
