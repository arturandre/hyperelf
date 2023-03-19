#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

nohup python ${script_full_path}/comparisons.py \
${script_full_path}/trees_only_unknown_confignh.json \
/scratch/arturao/hyperelf/outputs/combined \
all_preds_tunh.csv \
images_tunh \
--copy-images > comp_tunh.out & tail -f comp_tunh.out