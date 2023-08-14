#!/bin/bash

# Here we train and test networks from scratch on
# ImageNet2012

script_name=$0
script_full_path=$(dirname "$0")

dbname="trees"

expnames=("exp191")
networknames=("mobilenetv3")
networknameselyx=("MobileNetV3LargeElyx")
datasetnames=("TreesLocatingTest3k")
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
        echo "Starting experiment ${expname} on ${datasetname}"
        python ${script_full_path}/../general_experiment.py \
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
        --epochs "40" \
        --loss "nll" \
        --cweights "0.5,0.1,0.2,0.2" \
        --lr "0.5" \
        --gamma "0.7" \
        --log-interval "10" \
        --silence \
        > ${expname}${expnamesuffix}.out
    done
done

#        --custom-disagreement-csv "/scratch/arturao/hyperelf/outputs/disagreements-e68-74-nh/report_disagreements_train.csv" \
#        --custom-disagreement-threshold "5" \
