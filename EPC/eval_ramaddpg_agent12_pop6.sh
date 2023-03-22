#!/bin/sh
exp_name="ramaddpg_noise3_pop6"
noise=3

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=12 \
    --num-food=12 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-2/seed-0" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent12_seed0.log & disown 

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=12 \
    --num-food=12 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-2/seed-1" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent12_seed1.log & disown 

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=12 \
    --num-food=12 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-2/seed-2" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent12_seed2.log & disown 

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=12 \
    --num-food=12 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-2/seed-3" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent12_seed3.log & disown 

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=12 \
    --num-food=12 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-2/seed-4" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent12_seed4.log & disown 

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=12 \
    --num-food=12 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-2/seed-5" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent12_seed5.log & disown 