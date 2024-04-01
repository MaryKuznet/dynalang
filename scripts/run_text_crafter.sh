#! /bin/bash

type_data=$1
mode=$2
enc=$3
name=$4
device=$5
seed=$6

shift
shift
shift
shift
shift
shift

export CUDA_VISIBLE_DEVICES=$device; python dynalang/train.py \
  --run.script train \
  --logdir ~/logdir/crafter/$name \
  --use_wandb True \
  --task textcrafter_$type_data\
  --envs.amount 1 \
  --env.textcrafter.mode $mode \
  --env.textcrafter.enc $enc \
  --seed $seed \
  --batch_size 16 \
  --batch_length 256 \
  --configs textcrafter \
  "$@"