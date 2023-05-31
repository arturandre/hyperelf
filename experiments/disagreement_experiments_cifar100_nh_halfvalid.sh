#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

# This bash script is an example of use for the 
# python script of same name.

nohup python ${script_full_path}/disagreement_experiments.py \
"/scratch/arturao/hyperelf/outputs" \
"exp100nh,exp101nh,exp102nh,exp103nh,exp104nh,exp105nh" \
--save-disagreements \
--dataset-name-clean "CIFAR100HalfValid" \
--dataset-name-unknown "CIFAR100Half" \
--output-path "/scratch/arturao/hyperelf/outputs/disagreements-e100-105-nh" &

#--save-dataset \
