#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

# This bash script is an example of use for the 
# python script of same name.

nohup python ${script_full_path}/disagreement_experiments.py \
"/scratch/arturao/hyperelf/outputs" \
"--dataset-name-unknown" "ImageNetO" \
"exp143nh,exp144nh,exp145nh,exp146nh,exp147nh,exp148nh,exp149nh,exp150nh" \
--output-path "/scratch/arturao/hyperelf/outputs/disagreements-e143-150-nh" &
