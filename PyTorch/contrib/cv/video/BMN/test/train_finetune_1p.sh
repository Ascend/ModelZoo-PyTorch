#!/bin/bash
data_path=/opt/npu/
pth_path=./checkpoint/BMN_best.pth.tar
arch=`uname -m`
echo $arch

for para in $*
do
    if [[ $para == --data_path* ]]; then
        data_path=`echo ${para#*=}`
    fi
done

for para in $*
do
    if [[ $para == --pth_path* ]]; then
        pth_path=`echo ${para#*=}`
    fi
done

python3 -u -m torch.distributed.launch --nproc_per_node=1 main_8p.py --mode train --finetune=1 --data_path=${data_path} --pth_path=${pth_path} --train_epochs 1 --batch_size 16 --training_lr 1.5e-3 --DeviceID 4 --world_size 1 --is_distributed 1 > train_finetune_1p.log 2>&1 &
