#!/bin/bash

#cd ../..

# custom config
DATA="/media/data2/zhangzherui/data/DATA"
TRAINER=PromptSRC_KDPL

CFG=vit_b32_c2_ep20_batch4_4+4ctx_cross_datasets
SHOTS=16
LOADEP=25


my_root="./output/base2new/train_base/imagenet_off/shots_16/promptsrc_off/vit_b32_c2_ep20_batch4_4+4ctx_cross_datasets/"
output_root="./output/base2new/test_base/imagenet_off/shots_16/promptsrc_off/vit_b32_c2_ep20_batch4_4+4ctx_cross_datasets/"


CLASS_AGNOSTIC=False # whether to use class-agnostic KDPL (KDPL-CA) or not

# unccoment to run CROSS-DOMAIN
#DATASETS=(imagenetv2 imagenet_r imagenet_a imagenet_sketch)
DATASETS=(imagenet)

# unccoment to run CROSS-DATASET
#DATASETS=(oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet)
#DATASETS=(imagenet)
mytime=15:34:40
for DATASET in "${DATASETS[@]}"; do
    for SEED in 1 2 3; do

#        if [ "$CLASS_AGNOSTIC" = True ]; then # KDPL-CA
#          DIR=output/cross_domain_and_datasets/test/${DATASET}/shots_${SHOTS}/${TRAINER}-CA/${CFG}/seed${SEED}
#         else # KDPL
#          DIR=output/cross_domain_and_datasets/test/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
#        fi


        MODEL_DIR=${my_root}/${mytime}/seed${SEED}
        DIR=${output_root}/${mytime}/seed${SEED}
        echo "Evaluating model"
        CUDA_VISIBLE_DEVICES=4 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR}  \
        --load-epoch ${LOADEP} \
        --eval-only

    done
done