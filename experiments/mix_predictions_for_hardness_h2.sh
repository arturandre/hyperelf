#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

python ${script_full_path}/mix_predictions_for_hardness.py \
/scratch/arturao/hyperelf/outputs/disagreements-e68-74-nh/report_disagreements_train.csv \
/scratch/arturao/hyperelf/outputs/disagreements-e106-111-nh/report_disagreements_train.csv \
/scratch/arturao/hyperelf/utils/indexes/cifar100_train_idx.npy \
/scratch/arturao/hyperelf/utils/indexes/cifar100_valid_idx.npy \
--output-filepath /scratch/arturao/hyperelf/outputs/mix_disagreements-e68-74-nh-e106-111-nh/report_disagreements_train.csv
