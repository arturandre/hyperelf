#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

# This bash script is an example of use for the 
# python script of same name.

nohup python ${script_full_path}/disagreement_experiments.py \
"/scratch/arturao/hyperelf/outputs" \
"exp068nh,exp069nh,exp070nh,exp072nh,exp073nh,exp074nh" \
--save-dataset \
--save-disagreements \
--dataset-name-clean "CIFAR100Half" \
--dataset-name-unknown "CIFAR100HalfValid" \
--output-path "/scratch/arturao/hyperelf/outputs/disagreements-e68-74-nh" &

