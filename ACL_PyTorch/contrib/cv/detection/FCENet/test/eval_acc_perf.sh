#! /bin/bash

rm -rf preprocessed_imgs
mkdir preprocessed_imgs
echo "start preprocess..."
python fcenet_preprocess.py ./mmocr/data/icdar2015/imgs/test/

python ./gen_dataset_info.py \
bin \
preprocessed_imgs \
fcenet_prep_bin.info \
1280 2272

source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo "start inference..."
rm -rf result
./tools/msame/out/msame \
--model fcenet.om \
--input preprocessed_imgs \
--output result \
--outfmt TXT
echo "start postprocess..."
rm boundary_results.txt
python fcenet_postprocess.py
echo "start evaluate metric..."
python eval.py \
./mmocr/configs/textdet/fcenet/fcenet_r50_fpn_1500e_icdar2015.py \
fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth \
--eval hmean-iou