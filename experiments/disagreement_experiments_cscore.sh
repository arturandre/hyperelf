#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

# This bash script is an example of use for the 
# python script of same name.

python ${script_full_path}/disagreement_experiments_cscore.py \
--report-disagreement "/scratch/arturao/hyperelf/outputs/disagreements-e54-60-nh/report_disagreements_train.csv" \
--dataset-path "/scratch/arturao/hyperelf/outputs/disagreements-e54-60-nh/dataset" \
--cscores-reference "/scratch/arturao/structural-regularity/cscores.npy" \
--index-file-clean "/scratch/arturao/hyperelf/utils/indexes/mnist_train_idx.npy" \
--index-file-unknown "/scratch/arturao/hyperelf/utils/indexes/mnist_valid_idx.npy"