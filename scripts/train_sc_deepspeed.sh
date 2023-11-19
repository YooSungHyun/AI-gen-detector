export OMP_NUM_THREADS=16
export WANDB_PROJECT="essays"
export WANDB_ENTITY="tadev"
export WANDB_NAME="[bart]test"
export HUGGINGFACE_HUB_CACHE="./.cache"
export HF_DATASETS_CACHE="./.cache"
deepspeed --include localhost:2,3 --master_port 61000 ./train_sc.py \
  --model_name_or_path "roberta-large" \
  --train_datasets_path "./data/concat_data/kfold_under_sample_99/train/0" \
  --eval_datasets_path "./data/concat_data/kfold_under_sample_99/eval/0" \
  --num_train_epochs 10 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing true \
  --learning_rate 3e-5 \
  --warmup_steps 500 \
  --logging_strategy steps \
  --logging_steps 30 \
  --evaluation_strategy steps \
  --eval_steps 120 \
  --save_strategy steps \
  --save_steps 120 \
  --save_total_limit 3 \
  --metric_for_best_model eval_roc_auc \
  --greater_is_better true \
  --seed 42 \
  --fp16 false \
  --group_by_length \
  --output_dir ./outputs/ \
  --report_to wandb \
  --deepspeed ./configs/zero2.json