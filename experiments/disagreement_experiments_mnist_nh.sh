#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

# This bash script is an example of use for the 
# python script of same name.

nohup python ${script_full_path}/disagreement_experiments.py \
"/scratch/arturao/hyperelf/outputs" \
"exp054nh,exp055nh,exp056nh,exp058nh,exp059nh,exp060nh" \
--dataset-name-clean "MNISTHalf" \
--dataset-name-unknown "MNISTHalfValid" \
--output-path "/scratch/arturao/hyperelf/outputs/disagreements-e54-60-nh" &

#--save-dataset \
#--save-disagreements \
