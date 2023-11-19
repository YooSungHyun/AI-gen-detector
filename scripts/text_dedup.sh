#!/bin/bash
python -m text_dedup.minhash \
   --path "./data/concat_data/only_external" \
   --local \
   --seed 42 \
   --cache_dir "./cache" \
   --output "./data/concat_data/ex_minhash_dedup_99" \
   --column "text" \
   --batch_size 1000 \
   --threshold 0.99