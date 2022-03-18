# Shape Robust Text Detection with Progressive Scale Expansion Network

## Requirements
* NPU配套的run包安装(C20B030)
* Python 3.7.5
* PyTorch（NPU版本）
* apex（NPU版本）
* pyclipper
* Polygon3
* opencv-python 3.4

## 1P
1. 编辑 train_1p.sh device-list(NPU设备号) addr(本机ip地址) data-dir(数据集目录) remark(备注信息)
2. 运行 sh train_1p.sh
```
python3 -W ignore train_8p_anycard.py \
    --lr 0.001\  
    --dist-backend 'hccl' \
    --rank 0  \
    --workers 32 \
    --multiprocessing-distributed \
    --world-size 1 \
    --batch_size 16 \
    --device 'npu' \
    --opt-level 'O2' \
    --loss-scale 64 \
    --addr='XX.XXX.XXX.XXX'  \  #修改本机ip地址
    --seed 16  \
    --n_epoch 600 \
    --data-dir '/home/w50015720/npu/PSENet_data' \ #修改数据集目录
    --port 8272 \
    --schedule 200 400 \
    --device-list '0' \ # 修改NPU设备号
    --remark 'test'  # 修改备注信息
```
## 8P
1. 编辑 train_8p.sh device-list(NPU设备号) addr(本机ip地址) data-dir(数据集目录) remark(备注信息)
2. 运行 sh train_8p.sh

```
python3 -W ignore train_8p_anycard.py \
    --lr 0.008\
    --dist-backend 'hccl' \
    --rank 0  \
    --workers 32 \
    --multiprocessing-distributed \
    --world-size 1 \
    --batch_size 32 \
    --device 'npu' \
    --opt-level 'O2' \
    --loss-scale 64 \
    --addr='XX.XXX.XXX.XXX' \ #修改本机ip地址
    --seed 16  \
    --n_epoch 600 \
    --data-dir '/home/data/' \ #修改数据集目录
    --port 8271 \
    --schedule 200 400 \
    --device-list '0,1,2,3,4,5,6,7' \  # 修改NPU设备号 8卡
    --remark 'npu8pbatch32lr8' # 修改备注信息
```

