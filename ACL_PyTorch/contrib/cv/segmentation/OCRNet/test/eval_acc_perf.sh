#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
echo batch_size=1

rm -rf ./result
mkdir result
python3 OCRNet_preprocess.py --src_path $1 --bin_file_path bs1_bin --batch_size 1

rm -rf ./result
mkdir result
./msame --model "om/ocrnet_optimize_bs1.om" --input "bs1_bin/imgs" --output "result" --outfmt TXT 

cd result/*

path=`pwd`
echo $path
cd ../..

python3 OCRNet_postprocess.py --bin_file_path bs1_bin --pred_path $path