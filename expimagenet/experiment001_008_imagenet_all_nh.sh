#!/bin/bash

# Here we train and test networks from scratch on
# ImageNet2012

script_name=$0
script_full_path=$(dirname "$0")

nepproject="vision-ime/elyximagenet"
nepapitoken="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NmZmOTAzYi02NTJkLTQ0MzUtOTYzYi1kYjVjZTVjYzc4MmMifQ=="

dbname="imagenet2012"

expnames=("exp008" "exp001" "exp002" "exp003" "exp004" "exp005" "exp006" "exp007" "exp007")
networknames=("densenet161" "resnet50" "mobilenetv2" "mobilenetv3" "efficientnet" "vgg16" "densenet121" "resnet152" "resnet152")
networknameselyx=("DenseNet161Elyx" "ResNet50Elyx" "MobileNetV2Elyx" "MobileNetV3LargeElyx" "EfficientNetB0Elyx" "VGG16Elyx" "DenseNet121Elyx" "ResNet152Elyx" "ResNet152Elyx")

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
        --output-folder "/scratch/arturao/hyperelf/outputsimagenet/${expname}${expnamesuffix}" \
        --log-file "${expname}${expnamesuffix}_${dbname}_${networkname}.log" \
        --load-model "/scratch/arturao/hyperelf/outputsimagenet/exp008nh/exp008nh_imagenet2012_densenet161.pt" \
        --save-model "${expname}${expnamesuffix}_${dbname}_${networkname}" \
        --network-name "${networknameelyx}" \
        --project-tags "ResolutionResized" \
        --elyx-head "${elyxhead}" \
        --dataset-name "${datasetname}" \
        --batch-size "64" \
        --test-batch-size "64" \
        --avoidsaveconfmat \
        --from-scratch \
        --epochs "1000" \
        --patience "91" \
        --loss "nll" \
        --lr "1.0" \
        --gamma "0.9" \
        --lr-step "30" \
        --log-interval "10" \
        --silence \
        --nep-project "${nepproject}" \
        --nep-api-token "${nepapitoken}" \
        > ${expname}${expnamesuffix}.out
        wait
    done
    wait
done

#        --custom-disagreement-csv "/scratch/arturao/hyperelf/outputs/disagreements-e68-74-nh/report_disagreements_train.csv" \
#        --custom-disagreement-threshold "5" \
