#!/bin/sh
exp_name="ramaddpg_noise3_pop6"
noise=3

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=6 \
    --num-food=6 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-1/seed-0" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent6_seed0.log & disown 

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=6 \
    --num-food=6 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-1/seed-1" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent6_seed1.log & disown 

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=6 \
    --num-food=6 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-1/seed-2" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent6_seed2.log & disown 

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=6 \
    --num-food=6 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-1/seed-3" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent6_seed3.log & disown 

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=6 \
    --num-food=6 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-1/seed-4" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent6_seed4.log & disown 

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=6 \
    --num-food=6 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-1/seed-5" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent6_seed5.log & disown 