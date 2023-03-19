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


# - This experiment is based on the script 005
# - * Here we try focal loss
# - Here we train on the cifar100 dataset only to check
#   the mobilenetv3.

#--log-it-folder "/scratch/arturao/hyperelf/outputs/${expname}_debug/it" \

expname="exp006"

nohup python ${script_full_path}/general_experiment.py \
--exp-name ${expname} \
--output-folder "/scratch/arturao/hyperelf/outputs/${expname}_debug" \
--log-file "${expname}_stl10_mobilenetv3_debug.log" \
--save-model "${expname}_stl10_mobnetv3large_debug" \
--network-name "MobileNetV3LargeElyx" \
--project-tags "Elyx2Reg,ElyxTrainAll,ElyxTest,ElyxTestNoGt,ResolutionResized" \
--elyx-head "2" \
--dataset-name "STL10" \
--batch-size "250" \
--test-batch-size "250" \
--epochs "20" \
--loss "focal" \
--lr "1.0" \
--gamma "0.7" \
--log-interval "10" \
> ${expname}_debug.out & tail -f ${expname}_debug.out

#--load-model "" \
#--only-test "" \
