#!/bin/bash

datasets_path="data/BSR/BSDS500/data/images/test"
batch_size=1

# usage
if [ $# -ne 4 ]
then
    echo "usage: bash test/eval_acc_gpu.sh datasets_path data/BSR/BSDS500/data/images/test batch_size 1"
else
    datasets_path=$2
    batch_size=$4
fi

python3.7 rcf_postprocess.py --model pth --imgs_dir ${datasets_path} \
--pth_path RCF-pytorch/RCFcheckpoint_epoch12.pth --pth_output data/pth_bs${batch_size}_out \
--batch_size ${batch_size} ${batch_size} --height 321 481 --width 481 321
if [ $? != 0 ]; then
    echo "post process fail!"
    exit -1
else
    echo "post process success!"
fi

echo "====pth accuracy data===="

cd edge_eval_python
cd cxx/src
# source build.sh
cd ../..
rm -rf ../data/examples_pth/rcf_bs${batch_size}_eval_result
mkdir -p ../data/examples_pth/rcf_bs${batch_size}_eval_result
python3.7 main.py --alg "RCF" --model_name_list "rcf" --result_dir ../data/pth_bs${batch_size}_out \
--save_dir ../data/examples_pth/rcf_bs${batch_size}_eval_result --gt_dir ../data/BSR/BSDS500/data/groundTruth/test \
--key pth_result --file_format .mat --workers -1
cd ..
if [ $? != 0 ]; then
    echo "fail!"
    exit -1
fi