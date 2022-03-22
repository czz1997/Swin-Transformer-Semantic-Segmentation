export ACDA_PATH=/home/qiyan/datarepo/datasets/ACDC

mkdir -p data/datasets
cd data/datasets

for keyword in "fog" "night" "rain" "snow"
do
  mkdir -p ACDC/$keyword/leftImg8bit
  ln -s $ACDA_PATH/rgb_anon/$keyword/* ACDC/$keyword/leftImg8bit

  mkdir -p ACDC/$keyword/gtFine
  ln -s $ACDA_PATH/gt/$keyword/* ACDC/$keyword/gtFine
done

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install "mmcv-full>=1.1.4,<=1.3.0" -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

cd ../..
pip install -e .

mkdir -p weights

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth -P weights

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth -P weights
