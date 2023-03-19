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


# - This experiment is based on the script 006
# * Its main purpose is to produce graphs (probability plots)
#   for each image at each intermediate outputs.

expname="exp007"

nohup python ${script_full_path}/general_experiment.py \
--exp-name ${expname} \
--output-folder "/scratch/arturao/hyperelf/outputs/${expname}" \
--log-it-folder "/scratch/arturao/hyperelf/outputs/${expname}/it" \
--log-file "${expname}_stl10_mobilenetv3.log" \
--network-name "MobileNetV3LargeElyx" \
--project-tags "Elyx2Reg,ElyxTrainAll,ElyxTest,ElyxTestNoGt,ResolutionResized" \
--elyx-head "mobnetv3l" \
--dataset-name "STL10" \
--batch-size "1" \
--test-batch-size "1" \
--epochs "20" \
--loss "focal" \
--lr "1.0" \
--gamma "0.7" \
--log-interval "10" \
--load-model "/scratch/arturao/hyperelf/outputs/uncategorized/exp006_stl10_mobnetv3large.pt" \
--only-test \
> ${expname}.out & tail -f ${expname}.out
