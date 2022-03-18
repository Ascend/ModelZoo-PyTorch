#!/bin/bash

input_path="./input"
image_path="./results"
num_input=200

if [ -d ${input_path} ]; then
  rm -rf ${input_path}
fi

python stylegan2-ada-pytorch_preprocess.py --batch_size=1 --num_input=${num_input}
#python stylegan2-ada-pytorch_preprocess.py --batch_size=16 --num_input=${num_input}

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
#mkdir -p dump/bs16

./msame --model "./G_ema_om_bs1.om" --input "./input/bs1" --output "./dump/bs1" >> dump/log.txt
#./msame --model "./G_ema_om_bs16.om" --input "./input/bs16" --output "./dump/bs16"

if [ -d "./dump/bs1" ]; then
  echo "inference with om success"
else
  echo "failed"
  exit -1
fi

bin_path_bs1="dump/bs1/`ls dump/bs1 | tail -1`"
#bin_path_bs16="dump/bs16/`ls dump/bs16 | tail -1`"
if [ -d ${image_path}]; then
  rm -r ${image_path}
fi

python stylegan2-ada-pytorch_postprocess.py --bin_path=${bin_path_bs1} --image_path=${image_path}

if [ $? != 0 ]; then
  echo "postprocess failed"
  exit -1
fi
echo "postprocess success"