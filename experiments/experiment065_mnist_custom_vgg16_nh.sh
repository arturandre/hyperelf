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

# Here the resnet50 will be trained
# on MNIST to predict ambiguous images from MNIST


# For some reason the script for multiple experiments (e.g. with/without Early exits)
# never runs the last experiment. So this is a workaround to run the last experiment.

dbname="mnist"

#expnames=("exp036" "exp037" "exp038")
#networknames=("resnet152" "resnet50" "efficientnet")
#networknameselyx=("ResNet152Elyx" "ResNet50Elyx" "EfficientNetB0Elyx")
expnames=("exp065")
networknames=("vgg16")
networknameselyx=("VGG16Elyx")
datasetnames=("MNISTCustom")
elyxheads=("None" )
suffixes=("nh")

for i in "${!expnames[@]}"; do
    expname=${expnames[i]}
    networkname=${networknames[i]}
    networknameelyx=${networknameselyx[i]}
    for j in "${!suffixes[@]}"; do
        elyxhead=${elyxheads[j]}
        expnamesuffix=${suffixes[j]}
        datasetname=${datasetnames[j]}
        nohup python ${script_full_path}/general_experiment.py \
        --exp-name ${expname}${expnamesuffix} \
        --output-folder "/scratch/arturao/hyperelf/outputs/${expname}${expnamesuffix}" \
        --log-file "${expname}${expnamesuffix}_${dbname}_${networkname}.log" \
        --save-model "${expname}${expnamesuffix}_${dbname}_${networkname}" \
        --network-name "${networknameelyx}" \
        --project-tags "Elyx2Reg,ElyxTrainAll,ElyxTest,ElyxTestNoGt,ResolutionResized" \
        --elyx-head "${elyxhead}" \
        --dataset-name "${datasetname}" \
        --custom-disagreement-csv "/scratch/arturao/hyperelf/outputs/disagreements-e54-60-nh/report_disagreements_train.csv" \
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