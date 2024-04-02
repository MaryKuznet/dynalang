#! /bin/bash

type_data=$1
mode=$2
enc=$3
spec_seed=$4
name=$5
device=$6
seed=$7
checkpoint=$8

shift
shift
shift
shift
shift
shift
shift
shift

export CUDA_VISIBLE_DEVICES=$device; python dynalang/train.py \
  --run.script eval_only \
  --logdir ~/logdir/crafter/$name \
  --use_wandb True \
  --task textcrafter_$type_data \
  --envs.amount 1 \
  --seed $seed \
  --encoder.mlp_keys token_embed$ \
  --decoder.mlp_keys token_embed$ \
  --batch_size 16 \
  --batch_length 256 \
  --run.log_every 1\
  --run.from_checkpoint ~/logdir/crafter/$checkpoint/checkpoint.ckpt \
  --run.save_frames_to ~/logdir/crafter/test_$name \
  --env.textcrafter.mode $mode \
  --env.textcrafter.enc $enc \
  --env.textcrafter.spec_seed $spec_seed \
  --configs textcrafter \
  "$@"