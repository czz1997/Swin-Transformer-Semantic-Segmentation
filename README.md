# Towards Domain Generalized Segmentation with Transformer

This repo contains the code of Team 11's group project in the 2021/22 Winter Term2 course [EECE571F](https://lrjconan.github.io/DL-structures/) at UBC.

Team 11 | Group member: Qi Yan, Zhongze Chen, Menghong Huang, Rui Yao

## Get started
### Setup python environment
```bash
python -m venv venveece571f
source venveece571f/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install "mmcv-full>=1.1.4,<=1.3.0" -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

pip install -e .
pip install gdown
```

### Setup dataset
Please visit the [ACDC official website](https://acdc.vision.ee.ethz.ch/download) to download the original dataset.
After that, one could use our helper script to do the leave-one-domain-out preprocessing.
```bash
export ACDA_PATH=YOUR_PATH/ACDC  # specify the ACDC path
bash link_acdc_dataset.sh
```
The data structure after preprocessing is as follows:
<details>
<summary><b><u>Dataset structure after preprocessing</u></b></summary>

```bash
$ tree data/datasets/ -L 3
data/datasets/
├── ACDC_nouveau
│   ├── fog
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   └── val
│   │   └── leftImg8bit
│   │       ├── train
│   │       └── val
│   ├── night
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   └── val
│   │   └── leftImg8bit
│   │       ├── train
│   │       └── val
│   ├── rain
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   └── val
│   │   └── leftImg8bit
│   │       ├── train
│   │       └── val
│   └── snow
│       ├── gtFine
│       │   ├── train
│       │   └── val
│       └── leftImg8bit
│           ├── train
│           └── val
└── ACDC_vanilla
    ├── fog
    ├── night
    ├── rain
    └── snow
```
</details>

We use the data at `ACDC_nouveau` directory for our project.
Please refer to the `train` and `val` folders for the training and validation sets.
The `leftImg8bit` and `gtFine` folders contain the RGB images and the annotation respectively.

Note: the `ACDC_vanilla` folder is kept for internal inspection only and not used for training/testing.


## Inference
```bash
# download our checkpoint weights, based on style mixing + supervised contrastive loss
mkdir -p weights/ckpt
# fog 
gdown https://drive.google.com/uc?id=1_xxMH-l60IqCgjU3b0JgOc_OkHAM_76J -O weights/ckpt/fog.pth
# night
gdown https://drive.google.com/uc?id=1x8IIqUvl8rdYHYhM7FdcvMtCQvAU1ODG -O weights/ckpt/night.pth
# rain
gdown https://drive.google.com/uc?id=1-jrI584iyLZZtMJq7n2ZvSHXRIK6QkQs -O weights/ckpt/rain.pth
# snow
gdown https://drive.google.com/uc?id=1HcrB3k0-R5CGPgGwQ-XRzqcRZdJKFSGf -O weights/ckpt/snow.pth

# single-gpu testing
export DOMAIN=fog
export CONFIG_FILE=configs/swin_mixcon/upernet_swin_small_patch4_window7_512x512_80k_ACDC_nouveau_${DOMAIN}.py
export SEG_CKPT=weights/ckpt/${DOMAIN}.pth
# save qualitative results at output/$DOMAIN 
python tools/test.py $CONFIG_FILE $SEG_CKPT --eval mIoU --show-dir output/$DOMAIN
```
You may also check out our [colab notebook](https://colab.research.google.com/drive/18hrKdvIg4DZGNiIgDRnZtuPf8oPaw1Bh?usp=sharing) for model inference results.

## Training
### Download pretrained weights
```bash
mkdir -p weights

# backbone weights pretrained with classification on ImageNet dataset
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth -P weights
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth -P weights
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth -P weights

# backbone weights pretrained with SSL on ImageNet dataset
# simmim on base backbone
gdown https://drive.google.com/uc?id=15zENvGjHlM71uKQ3d2FbljWPubtrPtjl -O weights
# moby on tiny backbone
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_deit_small_300ep_pretrained.pth -P weights
```
### Go training
```bash
# run the SwinTF baseline
export DATASET=fog
export BACKBONE=small
bash run_acdc_nouveau_vanilla_80k.sh $DATASET $BACKBONE

# run the SwinTF + mixing style + contrastive loss (ours)
bash run_acdc_nouveau_mixcon_80k.sh $DATASET $BACKBONE

# run the SwinTF + simmim SSL + mixing style + contrastive loss (ours)
# the simmim SSL weight is pretrained on ImageNet dataset
export BACKBONE=base
bash run_acdc_nouveau_simmim_mixcon_80k.sh $DATASET $BACKBONE

# run the SwinTF + moby SSL + mixing style + contrastive loss (ours)
# the moby SSL weight is pretrained on ImageNet dataset
export BACKBONE=tiny
bash run_acdc_nouveau_moby_mixcon_80k.sh $DATASET $BACKBONE
```

**Notes:** 
- The default training requires at least 2 GPUs. You may get an error when running the training script using a single GPU. 
- Try changing the `SyncBN` keyword in the into `BN`. It is found at `configs/_base_/models/upernet_swin_mixcon.py` and `configs/_base_/models/upernet_swin_vanilla.py`. 

## Acknowledge
We thank the following open-source code repositories:
* [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
* [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
* [Swin-Transformer-Semantic-Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)
* [SimMIM](https://github.com/microsoft/SimMIM)
* [Transformer-SSL](https://github.com/SwinTransformer/Transformer-SSL)
* [ContrastiveSeg](https://github.com/tfzhou/ContrastiveSeg)
