#!/bin/bash

# Aborted because doesn't fit in the GPU.

script_name=$0
script_full_path=$(dirname "$0")

dbname="trees"

#_reweight The script for weighting focal loss was changed

expnames=(
    "exp190_reweight_large_lr_gamma_fullres_256"
    "exp194_reweight_large_lr_gamma_fullres_256"
    "exp195_reweight_large_lr_gamma_fullres_256"
    "exp196_reweight_large_lr_gamma_fullres_256")
networknames=("mobilenetv3" "mobilenetv3" "mobilenetv3" "mobilenetv3")
networknameselyx=("MobileNetV3LargeElyx" "MobileNetV3LargeElyx" "MobileNetV3LargeElyx" "MobileNetV3LargeElyx")
datasetnames=("TreesLocatingTest3k" "TreesLocatingTest3k" "TreesLocatingTest3k" "TreesLocatingTest3k")
createpseudolabels=("TreesUnlabed40k" "TreesUnlabed40k" "TreesUnlabed40k" "TreesUnlabed40k")
pseudodatasets=(
    "None"
    "TreesUnlabed40k,/scratch/arturao/hyperelf/outputs/exp190_reweight_large_lr_gamma_fullres_256nh/probs_train_TreesUnlabed40k_outputs.npy"
    "TreesUnlabed40k,/scratch/arturao/hyperelf/outputs/exp194_reweight_large_lr_gamma_fullres_256nh/probs_train_TreesUnlabed40k_outputs.npy"
    "TreesUnlabed40k,/scratch/arturao/hyperelf/outputs/exp195_reweight_large_lr_gamma_fullres_256nh/probs_train_TreesUnlabed40k_outputs.npy")
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
    --fullres \
    --freeze-backbone \
    --batch-size "256" \
    --test-batch-size "256" \
    --epochs "1000" \
    --patience "91" \
    --loss "focal" \
    --cweights "0.5,0.1,0.2,0.2" \
    --lr "1.0" \
    --gamma "0.9" \
    --lr-step "30" \
    --log-interval "10" \
    --silence \
    > ${expname}${expnamesuffix}.out
    done
done

#        --custom-disagreement-csv "/scratch/arturao/hyperelf/outputs/disagreements-e68-74-nh/report_disagreements_train.csv" \
#        --custom-disagreement-threshold "5" \
