-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [Requirements](#requirements)
-   [数据集](#数据集)
-   [代码及路径解释](#代码及路径解释)
-   [Running the code](#running-the-code)
	
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Image super-resolution**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.07.12**

**框架（Framework）：Pytorch 1.4.0**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于残差密集网络的图像超分辨实现** 

<h2 id="概述.md">概述</h2>

提出了一种新的残差密集网络(RDN)来解决图像SR(super-resolution)中的这个问题。RDN可以充分利用了所有卷积层的层次特征。

- 参考论文：

    [[1] Y. Zhang, Y. Tian, Y. Kong, B. Zhong, and Y. Fu, “Residual Dense Network for Image Super-Resolution,” arXiv:1802.08797 [cs], Mar. 2018, Accessed: Apr. 04, 2022. 
](http://arxiv.org/abs/1802.08797) 

- 参考github仓库：  
https://github.com/puffnjackie/pytorch-super-resolution-implementations

## Requirements
- Python 3.7.5.
- Pytorch 1.4.0
- numpy
- Huawei Ascend


## 数据集
训练数据集DIV2K， 测试数据集Set5， 缩放因子4
```
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=kMZ5rC1dFBGYHI6Ycr96rcKSqlXL/5HAm77GvUS9QYBHiqXrq0cOgWg0GfQ1aIoVF3O9EBVZh0CFvbtT8TFi70VjXynqXKrZ09bmxrLQ+NTsZlnaTqSt8Op3yJT243YpNXnx1IHPE8nynkP88cjdKKDMBAaKeyfmY6kC6zpzYllco7lLeHyA9fpVpRHdB4FP6C2a2t91MG1gxYOGlvSwxI1l7f/WbEbqJ7qZFWRpPatC2VX5v6jY/Cq6/jiI1UdWNiSUvDe4rmdZajB7d2ZpLzPWjhMg3/epTIFGvNnjGmsYFbEDrnc4Uh8qqsq4786QIUycBGTAkGgoQBavhcJ6jDkidMLgJucTxd9rM3OLeFLj7Er9TyW3q7NpqywF1JWqqEMWs6MNF3nZO8ew6Aca84QK5Fb0BYfevWocsaDc6/t19f5xu06SZqU4SJyUeUco8V9tZ/wiuJpLPNdKup9++tDSJkKdrGBeuWU3m170DReO2Dz44DhTJw/8OcSVYD545FEkGxwx8hTFqdG/5tR2Unq4nPbKxnEsumskgkBxAZo=

提取码:
123456

*有效期至: 2023/01/08 07:50:42 GMT-08:00
数据集下载后放入代码目录即可
```

## 代码及路径解释

RDN  
└─  
  ├─README.md  
  ├─LICENSE    
  ├─ckpt    模型保存文件夹    
  ├─dataset       存放数据集文件夹  
  ├─test       test文件夹，存放脚本    
  ├─model 模型实现  
  ├─dataset.py  数据处理代码  
  ├─utils.py  工具类  
  ├─train.py  模型训练函数  
  ├─inference.py  模型推理函数  
  ├─train_npu.py  模型在NPU上训练函数
  ├─requirements.txt  所需包  
 

## Running the code
```
直接python train_npu.py即可
```

## 精度结果
在Set5数据集，Scale设置为2情况下：  
论文中精度：PSNR-32.47db  
GPU精度: PSNR-16.978db  
NPU迁移后: PSNR-17.35db  
NPU超过GPU。

### NPU部分结果输出
```
===> Epoch121: Loss: 0.029514
===> Epoch121: Loss: 0.035182
===> Epoch121: Loss: 0.029570
===> Epoch121: Loss: 0.034652
===> Epoch121: Loss: 0.029916
===> Epoch 121 Complete: Avg. Loss: 0.032026
40.880717277526855
=====> Training 121 epochs completed
=====> Testing 121 epochs
===> Testing # 121 epoch
===> Avg. PSNR: 17.355233 dB
=====> Testing 121 epochs completed
=====> lr scheduler activated in 121 epochs
=====> lr scheduler activated in 121 epochs completed
=====> Save checkpoint 121 epochs
ckpt/RDN/model_epoch_121.pth
Checkpoint saved to ckpt/RDN/model_epoch_121.pth
=====> Save checkpoint 121 epochs completed
=====> Training 122 epochs
===> Training # 122 epoch
===> Epoch122: Loss: 0.043487
===> Epoch122: Loss: 0.037076
```