#!/usr/bin/env bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3.7.5 pretreatment.py --input_path='/home/CASIA-B' --output_path='/root/CASIA-B-Pre/'
echo 'Step 1 finished!'

mkdir CASIA-B-bin
python3.7.5 -u test.py --iter=-1 --batch_size 1 --cache=True --pre_process=True
echo 'Step 2 finished!'

python3.7.5 gen_dataset_info.py bin CASIA-B-bin CASIA-B-bin.info 64 64
echo 'Step 3 finished!'
