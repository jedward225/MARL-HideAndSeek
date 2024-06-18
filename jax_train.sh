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
    --lr 1e-4 \
    --steps-per-update 40 \
    --num-bptt-chunks 2 \
    --num-minibatches 4 \
    --entropy-loss-coef 0.001 \
    --value-loss-coef 0.5 \
    --num-channels 512 \
    --pbt-ensemble-size 0 \
    --pbt-past-policies 0 \
    --num-hiders 2 \
    --num-seekers 2 \
    --profile-port 5000 \
    --bf16
    #--pbt-ensemble-size 16 \
    #--pbt-past-policies 32 \
