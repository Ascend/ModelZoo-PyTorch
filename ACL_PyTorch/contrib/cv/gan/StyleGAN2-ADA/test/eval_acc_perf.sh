#!/bin/bash

input_path="./input"
image_path="./results"
num_input=200

if [ -d ${input_path} ]; then
  rm -rf ${input_path}
fi

python stylegan2-ada-pytorch_preprocess.py --batch_size=1 --num_input=${num_input}

if [ $? != 0 ]; then
  echo "prepare input failed"
  exit -1
fi

source env.sh
if [ -d "./dump" ]; then
  rm -r ./dump
fi

mkdir -p dump/bs1
touch dump/log.txt

 python3 ./ais_infer_x86_64/ais_infer.py --model ../G_ema_om_bs1.om --input ./input/bs1/ --output ./dump/bs1/ >> dump/log.txt

if [ -d "./dump/bs1" ]; then
  echo "inference with om success"
else
  echo "failed"
  exit -1
fi

bin_path_bs1="dump/bs1/`ls dump/bs1 | tail -1`"
if [ -d ${image_path}]; then
  rm -r ${image_path}
fi

rm ./dump/bs1/*/summary.json

python stylegan2-ada-pytorch_postprocess.py --bin_path=${bin_path_bs1} --image_path=${image_path}

if [ $? != 0 ]; then
  echo "postprocess failed"
  exit -1
fi
echo "postprocess success"