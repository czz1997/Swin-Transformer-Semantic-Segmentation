#export ACDA_PATH=/home/qiyan/datarepo/datasets/ACDC

mkdir -p data/datasets
cd data/datasets

# create link for the vanilla ACDC data structure
for keyword in "fog" "night" "rain" "snow"
do
  mkdir -p ACDC_vanilla/$keyword/leftImg8bit
  ln -sf $ACDA_PATH/rgb_anon/$keyword/* ACDC_vanilla/$keyword/leftImg8bit

  mkdir -p ACDC_vanilla/$keyword/gtFine
  ln -sf $ACDA_PATH/gt/$keyword/* ACDC_vanilla/$keyword/gtFine
done

export ACDC_VANILLA_PATH=$(pwd)/ACDC_vanilla
# re-organize the four scenes into leave-one-out setting
# i.e., the training for "fog" is the train+val of data from "night", "rain" and "snow" scenes
# the validation for "fog" is its own train+val data
# all testing data is omitted

for keyword in "fog" "night" "rain" "snow"
do
  mkdir -p ACDC_nouveau/$keyword/leftImg8bit/train
  mkdir -p ACDC_nouveau/$keyword/leftImg8bit/val
  mkdir -p ACDC_nouveau/$keyword/gtFine/train
  mkdir -p ACDC_nouveau/$keyword/gtFine/val
  for key_inner in "fog" "night" "rain" "snow"
  do
    if [ "$key_inner" = "$keyword" ]; then
        # scene A's train+val now becomes scene A's val
        cp -L -srf $ACDC_VANILLA_PATH/$key_inner/leftImg8bit/train/* ACDC_nouveau/$keyword/leftImg8bit/val
        cp -L -srf $ACDC_VANILLA_PATH/$key_inner/leftImg8bit/val/* ACDC_nouveau/$keyword/leftImg8bit/val
        cp -L -srf $ACDC_VANILLA_PATH/$key_inner/gtFine/train/* ACDC_nouveau/$keyword/gtFine/val
        cp -L -srf $ACDC_VANILLA_PATH/$key_inner/gtFine/val/* ACDC_nouveau/$keyword/gtFine/val
    else
        # scene non-A's train+val now becomes scene A's train
        cp -L -srf $ACDC_VANILLA_PATH/$key_inner/leftImg8bit/train/* ACDC_nouveau/$keyword/leftImg8bit/train
        cp -L -srf $ACDC_VANILLA_PATH/$key_inner/leftImg8bit/val/* ACDC_nouveau/$keyword/leftImg8bit/train
        cp -L -srf $ACDC_VANILLA_PATH/$key_inner/gtFine/train/* ACDC_nouveau/$keyword/gtFine/train
        cp -L -srf $ACDC_VANILLA_PATH/$key_inner/gtFine/val/* ACDC_nouveau/$keyword/gtFine/train
    fi
  done

done
