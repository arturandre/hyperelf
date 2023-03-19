#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

nohup python ${script_full_path}/comparisons.py \
${script_full_path}/trees_no_unknown_confignh.json \
/scratch/arturao/hyperelf/outputs/combined \
all_preds_tnh.csv \
images_tnh \
--copy-images > comp_tnh.out & tail -f comp_tnh.out