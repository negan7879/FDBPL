#!/bin/bash

#cd ../..

# custom config
DATA="/media/data2/zhangzherui/data/DATA"
TRAINER=promptsrc_off

CFG=vit_b32_c2_ep20_batch4_4+4ctx_cross_datasets
SHOTS=16
current_time=$(date +"%H:%M:%S")
CLASS_AGNOSTIC=False # whether to use class-agnostic KDPL (KDPL-CA) or not

DATASET=imagenet_off
fruits=(3 4 5)
for SEED in 1 2 3; do

    if [ "$CLASS_AGNOSTIC" = True ]; then # KDPL-CA
          DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}-CA/${CFG}/${current_time}/seed${SEED}
        else # KDPL
          DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${current_time}/seed${SEED}
    fi


    let gpu=${fruits[${SEED}-1]}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        CUDA_VISIBLE_DEVICES=${gpu} python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.KDPL.CLASS_AGNOSTIC ${CLASS_AGNOSTIC} \
        DATASET.NUM_SHOTS ${SHOTS} &
    fi

done
