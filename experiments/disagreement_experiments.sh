#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

# This bash script is an example of use for the 
# python script of same name.

nohup python ${script_full_path}/disagreement_experiments.py \
"/scratch/arturao/hyperelf/outputs" \
"exp032,exp033,exp034,exp036,exp037,exp038" \
--output-path "/scratch/arturao/hyperelf/outputs/disagreements-e32-38" &
