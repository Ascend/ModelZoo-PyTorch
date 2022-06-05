#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
echo batch_size=1

rm -rf ./result
mkdir result
# 需要修改数据集地址
python OCRNet_preprocess.py --src_path /opt/npu/cityscapes/ --bin_file_path bs1_bin --batch_size 1

rm -rf ./result
mkdir result
./msame --model "om/ocrnet_optimize_bs1.om" --input "bs1_bin/imgs" --output "result" --outfmt TXT 

cd result/*

path=`pwd`
echo $path
cd ../..

python OCRNet_postprocess.py --bin_file_path bs1_bin --pred_path $path