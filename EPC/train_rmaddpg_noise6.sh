#!/bin/sh

CUDA_VISIBLE_DEVICES="" python -m maddpg_o.experiments.train_normal \
    --num-episodes=150000 \
    --num-good=6 \
    --num-food=6 \
    --good-policy=r-maddpg \
    --save-dir="./result/rmaddpg_noise6_seed1/agent6" \
    --save-rate=1000 \
    --train-rate=1000 \
    --n-cpu-per-agent=40 \
		--noise-std=6.0 \
		2>&1 | tee rmaddpg_noise6_agent6_seed1.log
