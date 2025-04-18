#!/bin/bash

ROOT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ -z $1 ]; then
    exit
fi

if [ -z $2 ]; then
    exit
fi

MADRONA_MWGPU_KERNEL_CACHE=${ROOT_DIR}/build/cache python ${ROOT_DIR}/scripts/jax_infer.py \
    --gpu-sim \
    --ckpt-path ${ROOT_DIR}/ckpts/$2 \
    --num-hiders 3 \
    --num-seekers 3 \
    --num-steps 3600 \
    --num-worlds $1 \
    --record-log ${ROOT_DIR}/build/record
    #--print-action-probs \
    #--bf16 \
