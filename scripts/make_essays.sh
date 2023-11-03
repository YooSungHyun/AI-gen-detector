#!/bin/bash
python3 1_make_essays.py \
    --api_config_path="./configs/openai_api_config.json" \
    --output_path="./data/outputs.jsonl" \
    --num_samples=4000