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

nepproject="vision-ime/elyxge"
nepapitoken="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NmZmOTAzYi02NTJkLTQ0MzUtOTYzYi1kYjVjZTVjYzc4MmMifQ=="

dbname="inaturalist2021mini"

# i's
expnames=("exp003")
networknames=("resnet50")
networknameselyx=("ResNet50Elyx")

# j's
datasetnames=("iNaturalist2021Mini")
elyxheads=("2")
suffixes=("")

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
        --output-folder "/scratch/arturao/hyperelf/outputsge/${expname}${expnamesuffix}" \
        --log-file "${expname}${expnamesuffix}_${dbname}_${networkname}.log" \
        --save-model "${expname}${expnamesuffix}_${dbname}_${networkname}" \
        --network-name "${networknameelyx}" \
        --project-tags "ResolutionResized" \
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
        --nep-project "${nepproject}" \
        --nep-api-token "${nepapitoken}" \
        > ${expname}${expnamesuffix}.out
        wait
    done
    wait
done

#        --custom-disagreement-csv "/scratch/arturao/hyperelf/outputs/disagreements-e68-74-nh/report_disagreements_train.csv" \
#        --custom-disagreement-threshold "5" \
