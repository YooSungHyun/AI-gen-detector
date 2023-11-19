#!/bin/bash
python -m text_dedup.minhash \
   --path "./data/concat_data/total_dataset" \
   --local \
   --seed 42 \
   --cache_dir "./cache" \
   --output "./data/concat_data/minhash_dedup_99" \
   --column "text" \
   --batch_size 1000 \
   --threshold 0.99