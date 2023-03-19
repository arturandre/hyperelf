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

echo -e "\nThis will hold the terminal, press CTRL+Z then type 'bg' and press ENTER.\n"

dbname="trees"

# i loop
expnames=("exp034" "exp035")
networknames=("densenet161" "vgg16")
networknameselyx=("DenseNet161Elyx" "VGG16Elyx")
#
# j loop
datasetnames=("TreesCustomNoUnknown" "TreesCustomNoUnknown" "TreesCustomUnknownPositive" "TreesCustomUnknownPositive" "TreesCustomUnknownNegative" "TreesCustomUnknownNegative")
elyxheads=("2" "None" "2" "None" "2" "None")
suffixes=("fullf" "fullfnh" "fullfup" "fullfnhup" "fullfun" "fullfnhun")

echo -e "\n\nSTART\n\n" >> nohup.out

for i in "${!expnames[@]}"; do
    echo "SLEEP i: ${i}" >> nohup.out
    for j in "${!suffixes[@]}"; do
        nohup echo "SLEEP j: ${j}" &
        sleep 1
        wait
    done
done