#!/bin/bash

# This one contains the baseline graph

script_name=$0
script_full_path=$(dirname "$0")

# This bash script is an example of use for the 
# python script of same name.

nohup python ${script_full_path}/fetch_neptune_to_report.py \
"HYPER-" \
"440-457,612-635" \
--suffix "baseline" &
