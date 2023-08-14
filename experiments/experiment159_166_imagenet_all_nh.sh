#!/bin/bash

# Here we only test pytorch pretrained weights on
# ImageNet-2012 test partition.

script_name=$0
script_full_path=$(dirname "$0")

dbname="imagenet2012"

expnames=("exp159" "exp160" "exp161" "exp162" "exp163" "exp164" "exp165" "exp166" "exp166")
networknames=("resnet50" "mobilenetv2" "mobilenetv3" "efficientnet" "vgg16" "densenet121" "densenet161" "resnet152" "resnet152")
networknameselyx=("ResNet50Elyx" "MobileNetV2Elyx" "MobileNetV3LargeElyx" "EfficientNetB0Elyx" "VGG16Elyx" "DenseNet121Elyx" "DenseNet161Elyx" "ResNet152Elyx" "ResNet152Elyx")
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
        nohup python ${script_full_path}/general_experiment.py \
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
        --epochs "1" \
        --only-test \
        --force-save \
        --keep-head \
        --loss "nll" \
        --lr "1.0" \
        --gamma "0.7" \
        --log-interval "10" \
        --silence \
        > ${expname}${expnamesuffix}.out &
        wait
    done
    wait
done

#        --custom-disagreement-csv "/scratch/arturao/hyperelf/outputs/disagreements-e68-74-nh/report_disagreements_train.csv" \
#        --custom-disagreement-threshold "5" \
