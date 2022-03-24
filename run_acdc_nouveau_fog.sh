export EXP_NAME="ACDC_nouveau_fog"
export CONFIG_FILE="configs/swin/upernet_swin_small_patch4_window7_512x512_80k_$EXP_NAME.py"
export PRETRAINED_MODEL_PATH="weights/swin_small_patch4_window7_224.pth"
export WORK_DIR="runs/${EXP_NAME}"
export NUM_GPUS=2

mkdir -p $WORK_DIR

# single gpu
#python tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR} --options model.pretrained=${PRETRAINED_MODEL_PATH} model.backbone.use_checkpoint=True

# multi gpu
bash tools/dist_train.sh ${CONFIG_FILE} $NUM_GPUS --work-dir ${WORK_DIR} --options model.pretrained=${PRETRAINED_MODEL_PATH} model.backbone.use_checkpoint=True
