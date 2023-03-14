#!/bin/sh
exp_name="rmaddpg_noise0"
noise=0

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=3 \
    --num-food=3 \
    --good-policy=r-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/agent3" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent3.log & disown 

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=6 \
    --num-food=6 \
    --good-policy=r-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/agent6" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent6.log & disown 

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=12 \
    --num-food=12 \
    --good-policy=r-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/agent12" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent12.log & disown 