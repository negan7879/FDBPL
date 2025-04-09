#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/dataset/folder"
TRAINER=VPT

CFG=vit_b32_c2_ep5_batch4_4
SHOTS=16
LOADEP=5
SUB=new

DATASETS=(oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet)

for DATASET in "${DATASETS[@]}"; do
    for SEED in 1 2 3; do

        COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
        DIR=output/base2new/test_${SUB}/${COMMON_DIR}

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
