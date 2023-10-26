#!/bin/bash

# Using the pseudo-labels generated by experiment
# 190_4 this will be a second-generation student.
# Third and fourth students will be generated too.

script_name=$0
script_full_path=$(dirname "$0")

dbname="trees"

#_reweight The script for weighting focal loss was changed

expnames=("exp190_outlier3_sharpen4_tophalf_noweighting_64")
networknames=("mobilenetv3")
networknameselyx=("MobileNetV3LargeElyx")
datasetnames=("TreesLocatingTest3kTernary")
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
    python ${script_full_path}/../general_experiment.py \
    --exp-name ${expname}${expnamesuffix} \
    --output-folder "/scratch/arturao/hyperelf/outputs/${expname}${expnamesuffix}" \
    --log-file "${expname}${expnamesuffix}_${dbname}_${networkname}.log" \
    --save-model "${expname}${expnamesuffix}_${dbname}_${networkname}" \
    --network-name "${networknameelyx}" \
    --project-tags "RandomUnknownBinary" \
    --elyx-head "${elyxhead}" \
    --dataset-name "${datasetname}" \
    --adjust-sharpness "4" \
    --tophalf \
    --batch-size "64" \
    --test-batch-size "64" \
    --epochs "1000" \
    --patience "91" \
    --loss "focal" \
    --lr "1.0" \
    --gamma "0.9" \
    --lr-step "30" \
    --log-interval "10" \
    --silence \
    > ${expname}${expnamesuffix}.out
    done
done

#        --custom-disagreement-csv "/scratch/arturao/hyperelf/outputs/disagreements-e68-74-nh/report_disagreements_train.csv" \
#        --custom-disagreement-threshold "5" \