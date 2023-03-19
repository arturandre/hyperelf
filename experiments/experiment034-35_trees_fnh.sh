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
# - experiment031_trees_densenet161.sh, 
# - experiment033_trees_mobilenetv3nh.sh,
# - experiment033_trees_mobilenetv3un.sh,
# - experiment033_trees_mobilenetv3unnh.sh,
# - experiment033_trees_mobilenetv3up.sh, and
# - experiment033_trees_mobilenetv3upnh.sh.
# The idea is to have all the experiments
# running from a single entry script, sequencially.

# For the fnh (half training) the idea is to
# use only half of the known samples to train the
# networks. Next these networks will be used to
# distinguish between the other half of known samples
# and the full set of unknown samples.
# The set of predicted known samples will
# be used together with the first half to train new networks.
# Two experiments will be conducted for unknown samples
# mistakengly predicted as known: they'll be assigned to the
# positive class, to the negative class.
# Finally, the networks trained with this larger set will be
# compared against the ones trained with all the known images.

dbname="trees"

expnames=("exp035")
networknames=("vgg16")
networknameselyx=("VGG16Elyx")
#expnames=("exp036")
#networknames=("resnet152")
#networknameselyx=("ResNet152Elyx")
datasetnames=("TreesHalfNoUnknown" "TreesHalfNoUnknown")
elyxheads=("2" "None")
suffixes=("f" "fnh")

for i in "${!expnames[@]}"; do
    expname=${expnames[i]}
    networkname=${networknames[i]}
    networknameelyx=${networknameselyx[i]}
    for j in "${!suffixes[@]}"; do
        elyxhead=${elyxheads[j]}
        expnamesuffix=${suffixes[j]}
        datasetname=${datasetnames[j]}
        echo "${expname} ${networkname} ${networknameelyx}" >> "${expname}${expnamesuffix}.out"
        echo "${elyxhead} ${expnamesuffix} ${datasetname}" >> "${expname}${expnamesuffix}.out"
        nohup python ${script_full_path}/general_experiment.py \
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
        >> "${expname}${expnamesuffix}.out" &
        wait
    done
    wait
done