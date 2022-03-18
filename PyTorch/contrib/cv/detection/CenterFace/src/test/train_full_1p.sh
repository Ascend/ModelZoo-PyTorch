#!/bin/bash
currentDir=$(cd "$(dirname "$0")";pwd)
echo $currentDir
dir=$(dirname $currentDir)
echo $dir
cd $dir
{
python3.7.5 main.py  --batch_size=32 --lr=5e-4 --lr_step='75,95'
python3.7.5 test_wider_face.py
dir1=$(dirname $dir)
echo $dir1
cd $dir1/evaluate
python3.7.5 setup.py build_ext --inplace
python3.7.5 evaluation.py --pred $dir1/output/widerface
} > CenterFace1p.log 2>&1 &