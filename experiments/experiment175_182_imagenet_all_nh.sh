#!/bin/bash

# Here we train and test networks from scratch on
# ImageNet2012

script_name=$0
script_full_path=$(dirname "$0")

dbname="imagenet2012"

#expnames=("exp175" "exp176" "exp177" "exp178" "exp179" "exp180" "exp181" "exp182" "exp182")
#networknames=("resnet50" "mobilenetv2" "mobilenetv3" "efficientnet" "vgg16" "densenet121" "densenet161" "resnet152" "resnet152")
#networknameselyx=("ResNet50Elyx" "MobileNetV2Elyx" "MobileNetV3LargeElyx" "EfficientNetB0Elyx" "VGG16Elyx" "DenseNet121Elyx" "DenseNet161Elyx" "ResNet152Elyx" "ResNet152Elyx")
expnames=("exp175" "exp176")
networknames=("resnet50" "mobilenetv2")
networknameselyx=("ResNet50Elyx" "MobileNetV2Elyx")

datasetnames=("ImageNet2012")
elyxheads=("None" )
suffixes=("nh")

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
        --output-folder "/scratch/arturao/hyperelf/outputs/${expname}${expnamesuffix}" \
        --log-file "${expname}${expnamesuffix}_${dbname}_${networkname}.log" \
        --save-model "${expname}${expnamesuffix}_${dbname}_${networkname}" \
        --network-name "${networknameelyx}" \
        --project-tags "ResolutionResized" \
        --elyx-head "${elyxhead}" \
        --dataset-name "${datasetname}" \
        --batch-size "80" \
        --test-batch-size "80" \
        --use-fsdp \
        --use-ddp \
        --world-size "2" \
        --from-scratch \
        --epochs "40" \
        --loss "nll" \
        --lr "0.5" \
        --gamma "0.7" \
        --log-interval "10" \
        --silence \
        > ${expname}${expnamesuffix}.out
    done
done

#        --custom-disagreement-csv "/scratch/arturao/hyperelf/outputs/disagreements-e68-74-nh/report_disagreements_train.csv" \
#        --custom-disagreement-threshold "5" \
