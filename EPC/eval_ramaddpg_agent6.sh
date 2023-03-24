#!/bin/sh
exp_name="ramaddpg_mix0.2_noise3"
noise=3

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=6 \
    --num-food=6 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-1/seed-0" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent6_seed0.log &

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=6 \
    --num-food=6 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-1/seed-1" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent6_seed1.log &

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=6 \
    --num-food=6 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-1/seed-2" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent6_seed2.log &