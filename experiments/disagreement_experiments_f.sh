#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

# This bash script is an example of use for the 
# python script of same name.

nohup python ${script_full_path}/disagreement_experiments.py \
"/scratch/arturao/hyperelf/outputs" \
"exp032f,exp033f,exp034f,exp036f,exp037f,exp038f" \
--output-path "/scratch/arturao/hyperelf/outputs/disagreements-e32-38-f" &
