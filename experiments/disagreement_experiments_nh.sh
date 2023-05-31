#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

# This bash script is an example of use for the 
# python script of same name.

nohup python ${script_full_path}/disagreement_experiments.py \
"/scratch/arturao/hyperelf/outputs" \
"exp032nh,exp033nh,exp034nh,exp036nh,exp037nh,exp038nh" \
--dataset-name-clean "TreesNoUnknown"
--dataset-name-unknown "TreesUnknownTestRev"
--output-path "/scratch/arturao/hyperelf/outputs/disagreements-e32-38-nh" &
