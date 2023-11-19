#!/bin/bash
NUM_GPU=2
GPU_IDS="0,1"
export OMP_NUM_THREADS=8
export WANDB_PROJECT=""
export WANDB_ENTITY=""
export WANDB_NAME=""
export HUGGINGFACE_HUB_CACHE="./.cache"
export HF_DATASETS_CACHE="./.cache"
model_name_or_path="EleutherAI/polyglot-ko-3.8b"

CUDA_VISIBLE_DEVICES=$GPU_IDS \
torchrun --nnodes=1 --nproc_per_node $NUM_GPU train_clm.py \
    --output_dir "outputs/" \
    --model_name_or_path "${model_name_or_path}" \
    --max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --save_total_limit 1 \
    --report_to wandb \
    --remove_unused_columns False \
    --dataloader_num_workers 4 \
    --ddp_find_unused_parameters False \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpoint \
    --optim adafactor \
    --fp16 