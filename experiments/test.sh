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

dbname="cifar100"

#expnames=("exp036" "exp037" "exp038")
#networknames=("resnet152" "resnet50" "efficientnet")
#networknameselyx=("ResNet152Elyx" "ResNet50Elyx" "EfficientNetB0Elyx")
expnames=("e1" "e2" "e3")
networknames=("nn1" "nn2" "nn3")
networknameselyx=("ne1" "ne2" "ne3")
datasetnames=("d1" "d2" "d3")
elyxheads=("h1" "h2" "h3" )
suffixes=("s1" "s2" "s3")

for i in "${!expnames[@]}"; do
    expname=${expnames[i]};
    networkname=${networknames[i]};
    networknameelyx=${networknameselyx[i]};
    echo "$networknameelyx";
    for j in "${!suffixes[@]}"; do
        elyxhead=${elyxheads[j]};
        expnamesuffix=${suffixes[j]};
        datasetname=${datasetnames[j]};
        nohup echo "$datasetname" &
        wait;
    done;
    wait;
done;