#!/bin/bash
export OMP_NUM_THREADS=8
export WANDB_PROJECT=""
export WANDB_ENTITY=""
export WANDB_NAME=""
export HUGGINGFACE_HUB_CACHE="./.cache"
export HF_DATASETS_CACHE="./.cache"
model_name_or_path="EleutherAI/polyglot-ko-12.8b"

CUDA_VISIBLE_DEVICES=$GPU_IDS \
deepspeed --include localhost:0,1,2,3 --master_port 61000 train_clm.py \
    --output_dir "outputs/" \
    --model_name_or_path "${model_name_or_path}" \
    --max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --save_total_limit 1 \
    --report_to wandb \
    --remove_unused_columns False \
    --dataloader_num_workers 0 \
    --ddp_find_unused_parameters False \
    --lr_scheduler_type "cosine" \
    --optim "adamw_torch" \
    --gradient_checkpointing True \
    --torch_compile True \
    --fp16 \
    --deepspeed ./ds_configs/zero2_fp16.json