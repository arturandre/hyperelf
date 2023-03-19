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


expname="exp038"
expnamesuffix=""
dbname="trees"
networkname="efficientnet"
networknameelyx="EfficientNetB0Elyx"

nohup python ${script_full_path}/general_experiment.py \
--exp-name ${expname}${expnamesuffix} \
--output-folder "/scratch/arturao/hyperelf/outputs/${expname}${expnamesuffix}" \
--log-file "${expname}${expnamesuffix}_${dbname}_${networkname}.log" \
--save-model "${expname}${expnamesuffix}_${dbname}_${networkname}" \
--network-name "${networknameelyx}" \
--project-tags "Elyx2Reg,ElyxTrainAll,ElyxTest,ElyxTestNoGt,ResolutionResized" \
--elyx-head "2" \
--dataset-name "TreesNoUnknown" \
--batch-size "80" \
--test-batch-size "80" \
--epochs "20" \
--loss "nll" \
--lr "1.0" \
--gamma "0.7" \
--log-interval "10" \
--silence \
> ${expname}${expnamesuffix}.out &&
expnamesuffix="nh" &&
nohup python ${script_full_path}/general_experiment.py \
--exp-name ${expname}${expnamesuffix} \
--output-folder "/scratch/arturao/hyperelf/outputs/${expname}${expnamesuffix}" \
--log-file "${expname}${expnamesuffix}_${dbname}_${networkname}.log" \
--save-model "${expname}${expnamesuffix}_${dbname}_${networkname}" \
--network-name "${networknameelyx}" \
--project-tags "Elyx2Reg,ElyxTrainAll,ElyxTest,ElyxTestNoGt,ResolutionResized" \
--elyx-head "None" \
--dataset-name "TreesNoUnknown" \
--batch-size "80" \
--test-batch-size "80" \
--epochs "20" \
--loss "nll" \
--lr "1.0" \
--gamma "0.7" \
--log-interval "10" \
--silence \
> ${expname}${expnamesuffix}.out &&
expnamesuffix="un" &&
nohup python ${script_full_path}/general_experiment.py \
--exp-name ${expname}${expnamesuffix} \
--output-folder "/scratch/arturao/hyperelf/outputs/${expname}${expnamesuffix}" \
--log-file "${expname}${expnamesuffix}_${dbname}_${networkname}.log" \
--save-model "${expname}${expnamesuffix}_${dbname}_${networkname}" \
--network-name "${networknameelyx}" \
--project-tags "Elyx2Reg,ElyxTrainAll,ElyxTest,ElyxTestNoGt,ResolutionResized" \
--elyx-head "2" \
--dataset-name "TreesUnknownNegative" \
--batch-size "80" \
--test-batch-size "80" \
--epochs "20" \
--loss "nll" \
--lr "1.0" \
--gamma "0.7" \
--log-interval "10" \
--silence \
> ${expname}${expnamesuffix}.out &&
expnamesuffix="unuh" &&
nohup python ${script_full_path}/general_experiment.py \
--exp-name ${expname}${expnamesuffix} \
--output-folder "/scratch/arturao/hyperelf/outputs/${expname}${expnamesuffix}" \
--log-file "${expname}${expnamesuffix}_${dbname}_${networkname}.log" \
--save-model "${expname}${expnamesuffix}_${dbname}_${networkname}" \
--network-name "${networknameelyx}" \
--project-tags "Elyx2Reg,ElyxTrainAll,ElyxTest,ElyxTestNoGt,ResolutionResized" \
--elyx-head "None" \
--dataset-name "TreesUnknownNegative" \
--batch-size "80" \
--test-batch-size "80" \
--epochs "20" \
--loss "nll" \
--lr "1.0" \
--gamma "0.7" \
--log-interval "10" \
--silence \
> ${expname}${expnamesuffix}.out &&
expnamesuffix="up" &&
nohup python ${script_full_path}/general_experiment.py \
--exp-name ${expname}${expnamesuffix} \
--output-folder "/scratch/arturao/hyperelf/outputs/${expname}${expnamesuffix}" \
--log-file "${expname}${expnamesuffix}_${dbname}_${networkname}.log" \
--save-model "${expname}${expnamesuffix}_${dbname}_${networkname}" \
--network-name "${networknameelyx}" \
--project-tags "Elyx2Reg,ElyxTrainAll,ElyxTest,ElyxTestNoGt,ResolutionResized" \
--elyx-head "2" \
--dataset-name "TreesUnknownPositive" \
--batch-size "80" \
--test-batch-size "80" \
--epochs "20" \
--loss "nll" \
--lr "1.0" \
--gamma "0.7" \
--log-interval "10" \
--silence \
> ${expname}${expnamesuffix}.out &&
expnamesuffix="upnh" &&
nohup python ${script_full_path}/general_experiment.py \
--exp-name ${expname}${expnamesuffix} \
--output-folder "/scratch/arturao/hyperelf/outputs/${expname}${expnamesuffix}" \
--log-file "${expname}${expnamesuffix}_${dbname}_${networkname}.log" \
--save-model "${expname}${expnamesuffix}_${dbname}_${networkname}" \
--network-name "${networknameelyx}" \
--project-tags "Elyx2Reg,ElyxTrainAll,ElyxTest,ElyxTestNoGt,ResolutionResized" \
--elyx-head "None" \
--dataset-name "TreesUnknownPositive" \
--batch-size "80" \
--test-batch-size "80" \
--epochs "20" \
--loss "nll" \
--lr "1.0" \
--gamma "0.7" \
--log-interval "10" \
--silence \
> ${expname}${expnamesuffix}.out &