#! /bin/bash

task=$1
name=$2
device=$3
seed=$4
checkpoint=$5

shift
shift
shift
shift
shift

export CUDA_VISIBLE_DEVICES=$device; python dynalang/train.py \
  --run.script eval_only \
  --logdir ~/logdir/crafter/$name \
  --use_wandb True \
  --task textcrafter_$task \
  --envs.amount 1 \
  --seed $seed \
  --encoder.mlp_keys token$ \
  --decoder.mlp_keys token$ \
  --decoder.vector_dist onehot \
  --batch_size 16 \
  --batch_length 256 \
  --run.train_ratio 32 \
  --run.from_checkpoint ~/logdir/crafter/$checkpoint/checkpoint.ckpt \
  --run.save_frames_to ~/logdir/crafter/test_$name \
  "$@"