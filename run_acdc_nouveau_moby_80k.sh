#!/bin/bash

# $DATASET must be "fog" "night" "rain" "snow"
DATASET=$1
if [ -z "$DATASET" ]
then
  echo "DATASET is empty"
  DATASET="fog"
else
  echo "DATASET is set"
fi
echo $DATASET


export EXP_NAME="ACDC_nouveau_$DATASET"
export CONFIG_FILE="configs/swin/upernet_swin_small_patch4_window7_512x512_80k_$EXP_NAME.py"
export PRETRAINED_MODEL_PATH="weights/moby_swin_small_patch4_window7_224_${DATASET}.pth"
export WORK_DIR="runs/${EXP_NAME}_adain_PT_moby_free"
export NUM_GPUS=2

mkdir -p $WORK_DIR

# single gpu
#python tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR} --options model.pretrained=${PRETRAINED_MODEL_PATH}

# multi gpu
bash tools/dist_train.sh ${CONFIG_FILE} $NUM_GPUS --work-dir ${WORK_DIR} --options model.pretrained=${PRETRAINED_MODEL_PATH}

