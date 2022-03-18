#!/bin/bash
currentDir=$(cd "$(dirname "$0")";pwd)
echo $currentDir
dir=$(dirname $currentDir)
echo $dir
cd $dir
{
python3.7.5 -m torch.distributed.launch --nproc_per_node=8 8p_npu_main.py --device_list='0,1,2,3,4,5,6,7' --world_size=8 --batch_size=16 --lr=2.5e-3 --lr_step='85,120'  --port='34577'
python3.7.5 test_wider_face.py
dir1=$(dirname $dir)
echo $dir1
cd $dir1/evaluate
python3.7.5 setup.py build_ext --inplace
python3.7.5 evaluation.py --pred $dir1/output/widerface
} > CenterFace8p.log 2>&1 &