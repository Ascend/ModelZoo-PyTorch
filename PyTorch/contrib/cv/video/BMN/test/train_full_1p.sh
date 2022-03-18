#!/bin/bash
data_path=/opt/npu/
arch=`uname -m`
echo $arch

for para in $*
do
    if [[ $para == --data_path* ]]; then
        data_path=`echo ${para#*=}`
    fi
done

python3.7 -u -m torch.distributed.launch --nproc_per_node=1 main_8p.py --mode full --data_path=${data_path} --batch_size 16 --training_lr 1.5e-3 --DeviceID 0 --world_size 1 --is_distributed 1 > train_full_1p.log 2>&1 &
