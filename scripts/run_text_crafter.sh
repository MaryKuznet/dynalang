#! /bin/bash
# task = datasettype_mode_encoder 
# datasettype: text instructions for text_crafter 
#              "MediumInstructions"
#              "HardInstructions"
#              "MixedMediumHardInstructions"
#              "Random"
# mode: self.mode of text_crafter environment
#       "Train"
#       "Test"
# encoder: ?
#         "New"
#         "Old"

task=$1 
name=$2
device=$3
seed=$4

shift
shift
shift
shift

export CUDA_VISIBLE_DEVICES=$device; python dynalang/train.py \
  --run.script train \
  --logdir ~/logdir/crafter/$name \
  --use_wandb True \
  --task textcrafter_$task \
  --envs.amount 1 \
  --seed $seed \
  --encoder.mlp_keys token_embed$ \
  --decoder.mlp_keys token_embed$ \
  --encoder.cnn_keys image$ \
  --decoder.cnn_keys image$ \
  --decoder.vector_dist onehot \
  --batch_size 16 \
  --batch_length 256 \
  --run.train_ratio 32 \
  --run.log_keys_max '^log_achievement_.*'\
  --load_model False \
  --log_every 1000 \
  "$@"
