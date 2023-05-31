#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

python ${script_full_path}/mix_predictions_for_hardness.py \
/scratch/arturao/hyperelf/outputs/disagreements-e68-74-nh/report_disagreements_train.csv \
/scratch/arturao/hyperelf/outputs/disagreements-e100-105-nh/report_disagreements_train.csv \
--output-filepath /scratch/arturao/hyperelf/outputs/mix_disagreements-e68-74-nh-e100-105-nh/report_disagreements_train.csv
