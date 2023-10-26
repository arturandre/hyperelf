#!/bin/bash

# Using the pseudo-labels generated by experiment
# 190_4 this will be a second-generation student.
# Third and fourth students will be generated too.

script_name=$0
script_full_path=$(dirname "$0")

dbname="trees"

#_reweight The script for weighting focal loss was changed

expnames=("exp190_reweight_sum" "exp194_reweight_sum" "exp195_reweight_sum" "exp196_reweight_sum")
networknames=("mobilenetv3" "mobilenetv3" "mobilenetv3" "mobilenetv3")
networknameselyx=("MobileNetV3LargeElyx" "MobileNetV3LargeElyx" "MobileNetV3LargeElyx" "MobileNetV3LargeElyx")
datasetnames=("TreesLocatingTest3k" "TreesLocatingTest3k" "TreesLocatingTest3k" "TreesLocatingTest3k")
createpseudolabels=("TreesUnlabed40k" "TreesUnlabed40k" "TreesUnlabed40k" "TreesUnlabed40k")
pseudodatasets=(
    "None"
    "TreesUnlabed40k,/scratch/arturao/hyperelf/outputs/exp190_reweight_sumnh/probs_train_TreesUnlabed40k_outputs.npy"
    "TreesUnlabed40k,/scratch/arturao/hyperelf/outputs/exp194_reweight_sumnh/probs_train_TreesUnlabed40k_outputs.npy"
    "TreesUnlabed40k,/scratch/arturao/hyperelf/outputs/exp195_reweight_sumnh/probs_train_TreesUnlabed40k_outputs.npy")
elyxheads=("None" "None" "None" "None")
suffixes=("nh" "nh" "nh" "nh")

for i in "${!expnames[@]}"; do
    expname=${expnames[i]}
    networkname=${networknames[i]}
    networknameelyx=${networknameselyx[i]}
    datasetname=${datasetnames[i]}
    createpseudolabel=${createpseudolabels[i]}
    pseudodataset=${pseudodatasets[i]}
    elyxhead=${elyxheads[i]}
    expnamesuffix=${suffixes[i]}
    echo "Starting experiment ${expname} on ${datasetname}"
    python ${script_full_path}/../general_experiment.py \
    --exp-name ${expname}${expnamesuffix} \
    --output-folder "/scratch/arturao/hyperelf/outputs/${expname}${expnamesuffix}" \
    --log-file "${expname}${expnamesuffix}_${dbname}_${networkname}.log" \
    --save-model "${expname}${expnamesuffix}_${dbname}_${networkname}" \
    --create-pseudo-label-datasets "${createpseudolabel}" \
    --pseudo-datasets "${pseudodataset}" \
    --network-name "${networknameelyx}" \
    --project-tags "ResolutionResized PseudoLabeled" \
    --elyx-head "${elyxhead}" \
    --dataset-name "${datasetname}" \
    --freeze-backbone \
    --batch-size "80" \
    --test-batch-size "80" \
    --epochs "1000" \
    --patience "20" \
    --loss "focal" \
    --cweights "0.5,0.1,0.2,0.2" \
    --lr "0.5" \
    --gamma "0.7" \
    --log-interval "10" \
    --silence \
    > ${expname}${expnamesuffix}.out
    done
done

#        --custom-disagreement-csv "/scratch/arturao/hyperelf/outputs/disagreements-e68-74-nh/report_disagreements_train.csv" \
#        --custom-disagreement-threshold "5" \
