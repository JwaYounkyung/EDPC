#!/bin/sh
exp_name="ramaddpg_qnature_mutation_noise0"
noise=0

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=3 \
    --num-food=3 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-0/seed-0" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent3_seed0.log & 

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=3 \
    --num-food=3 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-0/seed-1" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent3_seed1.log &

python maddpg_o/experiments/train_helper/train_helpers_eval.py \
    --num-good=3 \
    --num-food=3 \
    --good-policy=r-att-maddpg \
	--noise-std=$noise \
    --load-dir="./result/$exp_name/stage-0/seed-2" \
	--num-cpu=60 \
    --benchmark-iters=10000 \
    2>&1 | tee log/eval/${exp_name}_agent3_seed2.log &