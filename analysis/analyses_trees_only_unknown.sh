#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

nohup python ${script_full_path}/comparisons.py \
${script_full_path}/trees_only_unknown_config.json \
/scratch/arturao/hyperelf/outputs/combined \
all_preds_tu.csv \
images_tu \
--copy-images > comp_tu.out & tail -f comp_tu.out