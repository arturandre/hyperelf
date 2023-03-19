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
# (OnlyTest) - Only the test is executed (not the train) and
#   the number of epoches is forced to be 1.
# (TestBatch1) - Test batch has size one so each individual
#   output entropy can be assessed and recorded.
# (FromScratch) - Network trained from scratch.


# - This experiment is based on the script 013
# - Here we train on the iNaturalist2021 dataset, using the ResNet152.

#--log-it-folder "/scratch/arturao/hyperelf/outputs/${expname}_debug/it" \

expname="exp014"
dbname="inaturalist2021mini"
networkname="resnet152"

nohup python ${script_full_path}/general_experiment.py \
--exp-name ${expname} \
--output-folder "/scratch/arturao/hyperelf/outputs/${expname}" \
--log-file "${expname}_${dbname}_${networkname}.log" \
--save-model "${expname}_${dbname}_${networkname}" \
--network-name "ResNet152Elyx" \
--project-tags "Elyx2Reg,ElyxTrainAll,ElyxTest,ElyxTestNoGt,ResolutionResized" \
--elyx-head "2" \
--dataset-name "iNaturalist2021Mini" \
--batch-size "80" \
--test-batch-size "80" \
--epochs "20" \
--loss "nll" \
--lr "1.0" \
--gamma "0.7" \
--log-interval "10" \
> ${expname}.out & tail -f ${expname}.out

#--load-model "" \
#--only-test "" \