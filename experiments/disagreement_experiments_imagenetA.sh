#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

# This bash script is an example of use for the 
# python script of same name.

nohup python ${script_full_path}/disagreement_experiments.py \
"/scratch/arturao/hyperelf/outputs" \
"--dataset-name-unknown" "ImageNetA" \
"exp151nh,exp152nh,exp153nh,exp154nh,exp155nh,exp156nh,exp157nh,exp158nh" \
--output-path "/scratch/arturao/hyperelf/outputs/disagreements-e151-158-nh" &
