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

# - This experiment is based on the script 015
# - The results on Neptune.AI for exp14 and exp15 shows
#   that the pretrained ResNet50 have a better test accuracy
#   than ResNet152. This is surprising to me as ResNets were
#   supposed to learn residuals, and from my understanding
#   shallower ResNets should be 'subsets' of deeper ResNets.
#   The explanation could be that this assumption only holds
#   for networks trained from scratch. Thus here I'll train
#   the ResNet50 from scratch (without Elyx) and in the next
#   script I'll train the ResNet152 in the same conditions.
#   The results for experiments 18 and 19 should be compared
#   with those from experiments 14 and 16.

expname="exp018"
dbname="inaturalist2021mini"
networkname="resnet50"

nohup python ${script_full_path}/general_experiment.py \
--exp-name ${expname} \
--output-folder "/scratch/arturao/hyperelf/outputs/${expname}" \
--log-file "${expname}_${dbname}_${networkname}.log" \
--save-model "${expname}_${dbname}_${networkname}" \
--from-scratch \
--network-name "ResNet50Elyx" \
--project-tags "FromScratch,ResolutionResized" \
--elyx-head "None" \
--dataset-name "iNaturalist2021Mini" \
--batch-size "100" \
--test-batch-size "100" \
--epochs "20" \
--loss "nll" \
--lr "1.0" \
--gamma "0.7" \
--log-interval "10" \
> ${expname}.out & tail -f ${expname}.out

#--load-model "/scratch/arturao/hyperelf/outputs/uncategorized/inaturalist21mini_94cnn.pt" \
#--only-test \
#--log-it-folder "/scratch/arturao/hyperelf/outputs/${expname}/it" \