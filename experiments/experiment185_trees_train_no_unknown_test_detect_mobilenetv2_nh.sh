#!/bin/bash

# Here we train and test networks from scratch on
# ImageNet2012

script_name=$0
script_full_path=$(dirname "$0")

dbname="treesdetectingval"

expnames=("exp185")
networknames=("mobilenetv2")
networknameselyx=("MobileNetV2Elyx")
datasetnames=("TreesTestDetecting")
elyxheads=("None" )
suffixes=("nh")

for i in "${!expnames[@]}"; do
    expname=${expnames[i]}
    networkname=${networknames[i]}
    networknameelyx=${networknameselyx[i]}
    for j in "${!suffixes[@]}"; do
        datasetname=${datasetnames[j]}
        elyxhead=${elyxheads[j]}
        expnamesuffix=${suffixes[j]}
        nohup python ${script_full_path}/general_experiment.py \
        --exp-name ${expname}${expnamesuffix} \
        --output-folder "/scratch/arturao/hyperelf/outputs/${expname}${expnamesuffix}" \
        --log-file "${expname}${expnamesuffix}_${dbname}_${networkname}.log" \
        --save-model "${expname}${expnamesuffix}_${dbname}_${networkname}" \
        --network-name "${networknameelyx}" \
        --project-tags "ResolutionResized" \
        --elyx-head "${elyxhead}" \
        --dataset-name "${datasetname}" \
        --batch-size "80" \
        --test-batch-size "80" \
        --from-scratch \
        --epochs "20" \
        --loss "nll" \
        --lr "1.0" \
        --gamma "0.7" \
        --log-interval "10" \
        --silence \
        > ${expname}${expnamesuffix}.out &
        wait
    done
    wait
done

#        --custom-disagreement-csv "/scratch/arturao/hyperelf/outputs/disagreements-e68-74-nh/report_disagreements_train.csv" \
#        --custom-disagreement-threshold "5" \
