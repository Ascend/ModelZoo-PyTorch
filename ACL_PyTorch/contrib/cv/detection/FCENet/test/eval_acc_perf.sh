#! /bin/bash
#export profile_path=/home/zhangyifan/mmocr
export msame_path=/home/zhangyifan/tools/msame/out

rm -rf preprocessed_imgs
mkdir preprocessed_imgs

python ./mmocr/utils/fcenet_preprocess.py ./data/icdar2015/imgs/test/

python ./gen_dataset_info.py \
bin \
preprocessed_imgs \
fcenet_prep_bin.info \
1280 2272

source /usr/local/Ascend/ascend-toolkit/set_env.sh

rm -rf result
${msame_path}/msame \
--model fcenet.om \
--input preprocessed_imgs \
--output result \
--outfmt TXT

rm boundary_results.txt
python ./mmocr/models/textdet/postprocess/fcenet_postprocess.py

python ./tools/eval.py \
./configs/textdet/fcenet/fcenet_r50_fpn_1500e_icdar2015.py \
./fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth \
--eval hmean-iou