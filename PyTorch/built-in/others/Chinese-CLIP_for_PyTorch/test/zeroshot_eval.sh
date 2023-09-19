#!/bin/bash

# Usage: see example script below.
# bash run_scripts/zeroshot_eval.sh 0 \
#     ${path_to_dataset} ${dataset_name} \
#     ViT-B-16 RoBERTa-wwm-ext-base-chinese \
#     ${ckpt_path}

# only supports single-GPU inference
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}/test
fi

source ${test_path_dir}/env_npu.sh

path=${1}
dataset=cifar-100
datapath=${path}/datasets/${dataset}/test
savedir=${path}/save_predictions
vision_model=ViT-B-16 # ViT-B-16
text_model=RoBERTa-wwm-ext-base-chinese
resume=${path}/pretrained_weights/clip_cn_vit-b-16.pt
label_file=${path}/datasets/${dataset}/label_cn.txt


mkdir -p ${savedir}

python -u cn_clip/eval/zeroshot_evaluation.py \
    --datapath="${datapath}" \
    --label-file=${label_file} \
    --save-dir=${savedir} \
    --dataset=${dataset} \
    --img-batch-size=64 \
    --resume=${resume} \
    --vision-model=${vision_model} \
    --text-model=${text_model}
