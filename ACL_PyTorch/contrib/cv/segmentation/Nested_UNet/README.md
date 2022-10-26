# UNet++ Onnx模型端到端推理指导
- [UNet++ Onnx模型端到端推理指导](#unet-onnx模型端到端推理指导)
	- [1 模型概述](#1-模型概述)
		- [1.1 论文地址](#11-论文地址)
		- [1.2 代码地址](#12-代码地址)
	- [2 环境说明](#2-环境说明)
		- [2.1 深度学习框架](#21-深度学习框架)
		- [2.2 python第三方库](#22-python第三方库)
		- [2.3 获取ais_infer工具](#23-获取ais_infer工具)
	- [3 模型转换](#3-模型转换)
		- [3.1 pth转onnx模型](#31-pth转onnx模型)
		- [3.2 onnx转om模型](#32-onnx转om模型)
	- [4 数据集预处理](#4-数据集预处理)
		- [4.1 数据集获取](#41-数据集获取)
		- [4.2 数据集预处理](#42-数据集预处理)
		- [4.3 生成数据集信息文件](#43-生成数据集信息文件)
	- [5 离线推理](#5-离线推理)
		- [5.1 AisBench工具概述](#51-aisbench工具概述)
		- [5.2 离线推理](#52-离线推理)
	- [6 精度对比](#6-精度对比)
		- [6.1 离线推理IoU精度](#61-离线推理iou精度)
		- [6.2 开源IoU精度](#62-开源iou精度)
		- [6.3 精度对比](#63-精度对比)
	- [7 性能对比](#7-性能对比)
		- [7.1 npu性能数据](#71-npu性能数据)
		- [7.2 T4性能数据](#72-t4性能数据)
		- [7.3 性能对比](#73-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[UNet++论文](https://arxiv.org/abs/1807.10165)  

### 1.2 代码地址
[UNet++代码](https://github.com/4uiiurz1/pytorch-nested-unet)  
branch:master  
commit_id:557ea02f0b5d45ec171aae2282d2cd21562a633e  
备注：commit_id是指基于该次提交时的模型代码做推理，通常选择稳定版本的最后一次提交，或代码仓最新的一次提交  

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC2
pytorch >= 1.5.0
torchvision >= 0.6.0
onnx >= 1.7.0
```
实测环境中Torch的版本为1.5.0

### 2.2 python第三方库

```
numpy == 1.20.3
Pillow == 8.2.0
opencv-python == 4.5.2.54
albumentations == 0.5.2
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装


### 2.3 获取[ais_infer工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)

将工具编译后的压缩包放置在当前目录；解压工具包，安装工具压缩包中的whl文件； 
```
pip3 install aclruntime-0.01-cp37-cp37m-linux_xxx.whl
```

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.UNet++模型代码下载
```
git clone https://github.com/4uiiurz1/pytorch-nested-unet
```
2.原模型中resize采用双线性差值方法，影响模型在Ascend310P上的推理性能，需要改为最近邻方法。
```
cd pytorch-nested-unet
patch -p1 < ../nested_unet.diff
cd ..
```

3.编写pth2onnx脚本nested_unet_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 nested_unet_pth2onnx.py nested_unet.pth nested_unet.onnx
```

 **模型转换要点：**  
>此模型转换为onnx时需要将resize函数的模式修改为最近邻，参考nested_unet.diff文件

### 3.2 onnx转om模型

1.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.1.RC2 开发辅助工具指南 (推理) 01  
`${chip_name}`可通过 `npu-smi info` 指令查看，例: 310P3
![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=./nested_unet.onnx --input_format=NCHW --input_shape="actual_input_1:16,3,96,96" --output=nested_unet_bs16 --log=debug --soc_version=Ascend${chip_name}
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型将[2018 Data Science Bowl数据集](https://www.kaggle.com/c/data-science-bowl-2018)的训练集随机划分为训练集和验证集，为复现精度这里采用固定的验证集，验证集图像编号保存在val_ids.txt。

### 4.2 数据集预处理
1.执行原代码仓提供的数据集预处理脚本，并将处理后的数据集移动到当前目录下
```
cd pytorch-nested-unet
python3.7 preprocess_dsb2018.py
cp -r inputs/dsb2018_96/ ../
cp val_ids.txt ../
cd ..
```

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 preprocess_nested_unet.py ./dsb2018_96/images ./prep_dataset ./val_ids.txt
```

## 5 离线推理

-   **[AisBench工具概述](#51-AisBench工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 AisBench工具概述
AisBench推理工具，该工具包含前端和后端两部分。 后端基于c++开发，实现通用推理功能； 前端基于python开发，实现用户界面功能。
### 5.2 离线推理
1.执行离线推理
```
python3.7 /path/to/tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./nested_unet_bs1.om --input ./prep_dataset/ --output ./ais_results --outfmt BIN --batchsize=1
```
--model：模型地址  
--input：预处理完的数据集文件夹  
--output：推理结果保存地址  
--outfmt：推理结果保存格式  
--batchsize：模型batch size 默认为1 。当前推理模块根据模型输入和文件输出自动进行组batch。参数传递的  batchszie有且只用于结果吞吐率计算。请务必注意需要传入该值，以获取计算正确的吞吐率。  
输出结果默认保存在当前目录ais_results/X(X为执行推理的时间)，每个输入对应一个_X.bin文件的输出。


## 6 精度对比

-   **[离线推理IoU精度](#61-离线推理IoU精度)**  
-   **[开源IoU精度](#62-开源IoU精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理IoU精度

后处理统计IoU精度

调用postprocess_nested_unet.py脚本推理结果与语义分割真值进行比对，可以获得IoU精度数据。
```
python3.7 postprocess_nested_unet.py ./ais_results/2022_07_11-15_53_11/sumary.json ./dsb2018_96/masks/0/
```
第一个为AisBench输出目录，第二个为真值所在目录。  
查看输出结果：
```
IoU: 0.8385
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上。

### 6.2 开源IoU精度
[原代码仓公布精度](https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/README.md)
```
Model           IoU  
Nested U-Net    0.842  
```
### 6.3 精度对比
将得到的om离线模型推理IoU精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[T4性能数据](#72-T4性能数据)**  
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据
AisBench工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用AisBench纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，AisBench纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认AisBench工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用AisBench工具在整个数据集上推理得到bs1与bs16的性能数据为准。  

### 7.2 T4性能数据
在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2  
 
### 7.3 性能对比
|         | 310         | 310P3       | T4          | 310P3/310   | 310P3/T4     |
|---------|-------------|-------------|-------------|-------------|--------------|
| bs1     | 1674.950432 | 1681.600281 | 320.5865223 | 1.003970177 | 5.245386704  |
| bs4     | 1868.409806 | 2495.595994 | 400.8416683 | 1.335679135 | 6.22588965   |
| bs8     | 1845.83697  | 1907.807965 | 458.3922516 | 1.033573385 | 4.161955091  |
| bs16    | 1757.895362 | 1852.471243 | 549.9608379 | 1.053800632 | 3.368369374  |
| bs32    | 1724.390615 | 1796.602154 | 568.2170006 | 1.041876555 | 3.161824007  |
|         |             |             |             |             |              |
| 最优Batch | 1868.409806 | 2495.595994 | 568.2170006 | 1.335679135 | 4.391976993  |



310P单个device的吞吐率比310单卡的吞吐率大，故310P性能高于310性能，性能达标。  
对于batch1与batch16，310P性能均高于310性能1.2倍，该模型放在Benchmark/cv/segmentation目录下。  
 **性能优化：**  
>以上在310P上的结果为AOE优化后的性能。
因直接使用ATC导出模型已达标，所以不使用AOE进行性能优化。
