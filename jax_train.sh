#!/bin/bash

ROOT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ -z $1 ]; then
    exit
fi

mkdir -p ${ROOT_DIR}/ckpts
mkdir -p ${ROOT_DIR}/tb

rm -rf ${ROOT_DIR}/ckpts/$1
rm -rf ${ROOT_DIR}/tb/$1

CUDA_VISIBLE_DEVICES=0 MADRONA_MWGPU_KERNEL_CACHE=${ROOT_DIR}/build/cache python ${ROOT_DIR}/scripts/jax_train.py \
    --gpu-sim \
    --ckpt-dir ${ROOT_DIR}/ckpts \
    --tb-dir ${ROOT_DIR}/tb \
    --run-name $1 \
    --num-updates 100000 \
    --num-worlds 4096 \
    --lr 1e-3 \
    --steps-per-update 10 \
    --num-bptt-chunks 1 \
    --num-minibatches 1 \
    --num-epochs 4 \
    --entropy-loss-coef 0.01 \
    --value-loss-coef 0.5 \
    --num-channels 512 \
    --pbt-ensemble-size 2 \
    --pbt-past-policies 4 \
    --num-hiders 3 \
    --num-seekers 3 \
    --profile-port 5000 \
    --bf16
    # --num-epochs 16 # 60 / 5 
    #--steps-per-update 160 \
    #--num-bptt-chunks 16 \
    #
    #--pbt-ensemble-size 16 \
    #--pbt-past-policies 32 \
