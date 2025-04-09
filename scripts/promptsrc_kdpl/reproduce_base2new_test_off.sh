#!/bin/bash

#cd ../..

# custom config
DATA="/media/data2/zhangzherui/data/DATA"
TRAINER=PromptSRC_KDPL

CFG=vit_b32_c2_ep20_batch4_4+4ctx
SHOTS=16
LOADEP=25
SUB=new

CLASS_AGNOSTIC=False # whether to use class-agnostic KDPL (KDPL-CA) or not

#DATASETS=(oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet)
DATASETS=(imagenet)
mytime=10:37:13

my_root="./output/base2new/train_base/imagenet_off/shots_16/promptsrc_off/vit_b32_c2_ep20_batch4_4+4ctx/"
output_root="./output/base2new/test_base/imagenet_off/shots_16/promptsrc_off/vit_b32_c2_ep20_batch4_4+4ctx/"

# output/base2new/train_base/imagenet_off/shots_16/promptsrc_off/vit_b32_c2_ep20_batch4_4+4ctx/10:53:35

for DATASET in "${DATASETS[@]}"; do
    for SEED in 1 2 3; do

#        if [ "$CLASS_AGNOSTIC" = True ]; then # KDPL-CA
#          COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}-CA/${CFG}/seed${SEED}
#        else # KDPL
#          COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
#        fi

#        if [ "$CLASS_AGNOSTIC" = True ]; then # KDPL-CA
#          COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}-CA/${CFG}/${mytime}/seed${SEED}
#        else # KDPL
#          COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${mytime}/seed${SEED}
#        fi

#        MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
#        DIR=output/base2new/test_${SUB}/${COMMON_DIR}

        MODEL_DIR=${my_root}/${mytime}/seed${SEED}
        DIR=${output_root}/${mytime}/seed${SEED}

        echo "Evaluating model"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}

    done
done
