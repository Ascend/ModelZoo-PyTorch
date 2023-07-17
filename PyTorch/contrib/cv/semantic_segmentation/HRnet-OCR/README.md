# HRnet-OCR

## 模型简介

- 参考实现：

```
url=https:https://github.com/NVIDIA/semantic-segmentation
branch=master 
commit_id=7726b144c2cc0b8e09c67eabb78f027efdf3f0fa
```

- 模型原理：HRnet-OCR模型为图像分割网络，通过将注意力机制和多尺度预测的方法结合，实现了更快速的训练模型并保持更高精度。

##  Requirements

- CANN 5.0.3.1
- torch 1.5.0+ascend.post3.20210930
- apex 0.1+ascend.20210930
- tensor-fused-plugin 0.1+ascend
- te 0.4.0
- python 3.7.5
- runx 0.0.11
- torchvision 0.6.0

##  配置数据集路径

采用Cityscapes数据集

参考源码仓的方式获取数据集：https://github.com/NVIDIA/semantic-segmentation

获取数据集后需按照源代码仓Download/Prepare Data指示配置数据集路径

## 配置预训练模型

预训练模型权重在作者源代码仓中均已给出，配置路径请参照源代码仓Download Weights进行配置

## NPU 单卡训练命令

- 训练（注：训练结束后模型将自动打印评估结果）：

```
nohup bash test/train_full_1p.sh --data_path=./large_asset_dir/ &
```

- 性能：

```
nohup bash test/train_performance_1p.sh --data_path=./large_asset_dir/ &
```

## NPU 8卡训练命令

- 训练（注：训练结束后模型将自动打印评估结果）：

```
nohup bash test/train_full_8p.sh --data_path=./large_asset_dir/ &
```

- 性能：

```
nohup bash test/train_performance_8p.sh --data_path=./large_asset_dir/ &
```

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
