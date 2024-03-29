#!/bin/bash

script_name=$0
script_full_path=$(dirname "$0")

# Tags description:
# (ResolutionResized) - Resized resolution
# (ElyxTestNoGt) - During test, all the early exit criteria
#   are used, except the correctness one.
#   The motivation to exclude the correctness criterion is that
#   the test partition should assess how well the model generalizes
#   to unseen cases. Using the correctness criterion to select an
#   early exit (possible the last one) implies in having the
#   ground truth for unseen cases, thus making it no longer
#   an unseen case. 
# (ElyxTest) - At inference time the early exits can be used
#   to decrease the latency and resource usage. 
# (Elyx2Reg) - During training, the network is regularized
#   by early exits V2.
# (ElyxTrainAll) - During training, the early exit loss
#   is computed, but the whole network is trained.
# (TestBatch1) - Test batch has size one so each individual
#   output entropy can be assessed and recorded.
# (BackboneFrozen) - The backbone is frozen during fine-tuning,
#   where only the early exits and the final classification
#   head are trained.

# This script is based on:
# - experiment037_trees_resnet50_all.sh, 
# The idea is to experiment the techniques from 037
# applied to ImageNet2012Half, then use it
# to select images from the other Half
# and finally compare if the accuracy is better
# than training with the whole training
# dataset.

dbname="imagenet"

#expnames=("exp036" "exp037" "exp038")
#networknames=("resnet152" "resnet50" "efficientnet")
#networknameselyx=("ResNet152Elyx" "ResNet50Elyx" "EfficientNetB0Elyx")
expnames=("exp041")
networknames=("resnet152")
networknameselyx=("ResNet152Elyx")
datasetnames=("ImageNet2012Half" "ImageNet2012Half")
elyxheads=("2" "None" )
suffixes=("" "nh")

for i in "${!expnames[@]}"; do
    expname=${expnames[i]}
    networkname=${networknames[i]}
    networknameelyx=${networknameselyx[i]}
    for j in "${!suffixes[@]}"; do
        elyxhead=${elyxheads[j]}
        expnamesuffix=${suffixes[j]}
        datasetname=${datasetnames[j]}
        nohup python ${script_full_path}/general_experiment.py \
        --from-scratch \
        --exp-name ${expname}${expnamesuffix} \
        --output-folder "/scratch/arturao/hyperelf/outputs/${expname}${expnamesuffix}" \
        --log-file "${expname}${expnamesuffix}_${dbname}_${networkname}.log" \
        --save-model "${expname}${expnamesuffix}_${dbname}_${networkname}" \
        --network-name "${networknameelyx}" \
        --project-tags "Elyx2Reg,ElyxTrainAll,ElyxTest,ElyxTestNoGt,ResolutionResized" \
        --elyx-head "${elyxhead}" \
        --dataset-name "${datasetname}" \
        --batch-size "80" \
        --test-batch-size "80" \
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