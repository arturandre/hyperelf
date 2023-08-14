#!/bin/bash

# This one contains the graph for the
# networks trained only with half of the training set.

script_name=$0
script_full_path=$(dirname "$0")

# This bash script is an example of use for the 
# python script of same name.

nohup python ${script_full_path}/fetch_neptune_to_report.py \
"HYPER-" \
"636-649" \
--suffix "half" \
--output-folder /scratch/arturao/hyperelf/outputs/fetch_neptune/ &
