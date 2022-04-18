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


# $BACKBONE must be "tiny" "small" "base"
BACKBONE=$2
if [ -z "$BACKBONE" ]
then
  echo "BACKBONE is empty"
  BACKBONE="small"
else
  echo "BACKBONE is set"
fi
echo $BACKBONE


export EXP_NAME="ACDC_nouveau_${DATASET}"
export CONFIG_FILE="configs/swin_mixcon/upernet_swin_${BACKBONE}_patch4_window7_512x512_80k_$EXP_NAME.py"
export PRETRAINED_MODEL_PATH="weights/swin_${BACKBONE}_patch4_window7_224.pth"
export WORK_DIR="runs/${EXP_NAME}_swinTF_${BACKBONE}_mixcon"
export NUM_GPUS=4

mkdir -p $WORK_DIR

# single gpu
#python tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR} --options model.pretrained=${PRETRAINED_MODEL_PATH}

# multi gpu
bash tools/dist_train.sh ${CONFIG_FILE} $NUM_GPUS --work-dir ${WORK_DIR} --options model.pretrained=${PRETRAINED_MODEL_PATH}

