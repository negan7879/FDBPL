#!/bin/bash

#cd ../..

# custom config
#DATA="/path/to/dataset/folder"
#DATA="/media/data1/zhangzherui/data/DATA"
DATA="/media/data2/zhangzherui/data/DATA"

TRAINER=CoOp_KDPL

CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
CSC=False  # class-specific context (False or True)

# rn50_ctxv1, vit_b32_ctxv1
CFG=vit_b32_ctxv1
SHOTS=16
LOADEP=50
SUB=new


CLASS_AGNOSTIC=False # whether to use class-agnostic KDPL (KDPL-CA) or not

#DATASETS=(oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet)
DATASETS=(imagenet)


mytime=23:05:36


my_root="./output/base2new/train_base/imagenet_off/shots_16/CoOp_OFF/vit_b32_ctxv1"
output_root="./output/base2new/test_base/imagenet_off/shots_16/CoOp_OFF/vit_b32_ctxv1"



for DATASET in "${DATASETS[@]}"; do
    for SEED in 1 2 3; do

#        if [ "$CLASS_AGNOSTIC" = True ]; then # KDPL-CA
#          COMMON_DIR=${DATASET}/shots_${SHOTS}/CoOp_OFF${CFG}/${mytime}/seed${SEED}
#        else # KDPL
#          COMMON_DIR=${DATASET}/shots_${SHOTS}/CoOp_OFF/${CFG}/${mytime}/seed${SEED}
#        fi
        MODEL_DIR=${my_root}/${mytime}/seed${SEED}
        DIR=${output_root}/${mytime}/seed${SEED}
#        MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
#        DIR=output/base2new/test_${SUB}/${COMMON_DIR}

        echo "Evaluating model"
        CUDA_VISIBLE_DEVICES=2 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}

    done
done
