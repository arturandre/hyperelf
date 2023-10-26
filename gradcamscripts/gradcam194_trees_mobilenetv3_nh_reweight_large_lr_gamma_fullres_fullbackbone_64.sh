#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

dbname="trees"

expnames=("exp194_reweight_large_lr_gamma_fullres_fullbackbone_64")
networknames=("mobilenetv3")
networknameselyx=("MobileNetV3LargeElyx")
datasetnames=("TreesLocatingTest3k")
elyxheads=("None")
suffixes=("nh")

for i in "${!expnames[@]}"; do
    expname=${expnames[i]}
    networkname=${networknames[i]}
    networknameelyx=${networknameselyx[i]}
    datasetname=${datasetnames[i]}
    elyxhead=${elyxheads[i]}
    expnamesuffix=${suffixes[i]}
    echo "Starting experiment ${expname} on ${datasetname}"
    python ${script_full_path}/../general_gradcam.py \
    --output-folder "/scratch/arturao/hyperelf/outputs/${expname}${expnamesuffix}" \
    --log-file "${expname}${expnamesuffix}_${dbname}_${networkname}.log" \
    --load-model "/scratch/arturao/hyperelf/outputs/exp194_reweight_large_lr_gamma_fullres_fullbackbone_64nh/exp194_reweight_large_lr_gamma_fullres_fullbackbone_64nh_trees_mobilenetv3.pt" \
    --network-name "${networknameelyx}" \
    --elyx-head "${elyxhead}" \
    --dataset-name "${datasetname}" \
    --fullres \
    --batch-size "32" \
    --test-batch-size "32" \
    > ${expname}${expnamesuffix}.out
    done
done
