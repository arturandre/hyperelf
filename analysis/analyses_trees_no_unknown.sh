#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

nohup python ${script_full_path}/comparisons.py \
${script_full_path}/trees_no_unknown_config.json \
/scratch/arturao/hyperelf/outputs/combined \
all_preds_t.csv \
images_t \
--copy-images > comp_t.out & tail -f comp_t.out