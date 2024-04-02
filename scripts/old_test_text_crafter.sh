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
  --encoder.mlp_keys token$ \
  --decoder.mlp_keys token$ \
  --decoder.vector_dist onehot \
  --decoder.cnn_keys $^\
  --encoder.cnn_keys $^\
  --batch_size 16 \
  --batch_length 256 \
  --run.train_ratio 32 \
  --run.log_keys_max '^log_achievement_.*'\
  --run.log_every 1\
  --run.log_keys_video log_image \
  --run.from_checkpoint ~/logdir/crafter/$checkpoint/checkpoint.ckpt \
  --run.save_frames_to ~/logdir/crafter/test_$name \
  --env.textcrafter.mode $mode \
  --env.textcrafter.enc $enc \
  --env.textcrafter.spec_seed $spec_seed \
  --env.textcrafter.vis True \
  "$@"