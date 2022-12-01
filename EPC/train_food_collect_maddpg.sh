#!/bin/sh

CUDA_VISIBLE_DEVICES="" python -m maddpg_o.experiments.train_normal \
    --scenario=food_collect \
    --sight=100.0 \
    --num-episodes=50000 \
    --num-good=3 \
    --num-adversaries=0 \
    --num-food=3 \
    --num-units=32 \
    --checkpoint-rate=0 \
    --good-share-weights \
    --adv-share-weights \
    --good-policy=maddpg \
    --adv-policy=maddpg \

    --save-dir="./result/food_collect_maddpg_3" \
    --save-rate=100 \
    --train-rate=100 \
    --n-cpu-per-agent=24 \
    --stage-n-envs=25 \
    --timeout=0.03
