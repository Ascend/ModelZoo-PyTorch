# UNet++ Onnx模型端到端推理指导
- [UNet++ Onnx模型端到端推理指导](#unet-onnx模型端到端推理指导)
	- [1 模型概述](#1-模型概述)
		- [1.1 论文地址](#11-论文地址)
		- [1.2 代码地址](#12-代码地址)
	- [2 环境说明](#2-环境说明)
		- [2.1 深度学习框架](#21-深度学习框架)
		- [2.2 python第三方库](#22-python第三方库)
	- [3 模型转换](#3-模型转换)
		- [3.1 pth转onnx模型](#31-pth转onnx模型)
		- [3.2 onnx转om模型](#32-onnx转om模型)
	- [4 数据集预处理](#4-数据集预处理)
		- [4.1 数据集获取](#41-数据集获取)
		- [4.2 数据集预处理](#42-数据集预处理)
		- [4.3 生成数据集信息文件](#43-生成数据集信息文件)
	- [5 离线推理](#5-离线推理)
		- [5.1 benchmark工具概述](#51-benchmark工具概述)
		- [5.2 离线推理](#52-离线推理)
	- [6 精度对比](#6-精度对比)
		- [6.1 离线推理IoU精度](#61-离线推理iou精度)
		- [6.2 开源IoU精度](#62-开源iou精度)
		- [6.3 精度对比](#63-精度对比)
	- [7 性能对比](#7-性能对比)
		- [7.1 npu性能数据](#71-npu性能数据)



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
CANN 5.0.1
pytorch >= 1.5.0
torchvision >= 0.6.0
onnx >= 1.7.0
```

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

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.UNet++模型代码下载
```
git clone https://github.com/4uiiurz1/pytorch-nested-unet
```
2.原模型中resize采用双线性差值方法，影响模型在Ascend310上的推理性能，需要改为最近邻方法，并重新训练模型。将重新训练的模型移动到当前目录下，重命名为nested_unet.pth。
```
cd pytorch-nested-unet
patch -p1 < ../nested_unet.diff
python3.7 train.py --dataset dsb2018_96 --arch NestedUNet --loss LovaszHingeLoss --epochs 200  
cp models/dsb2018_96_NestedUNet_woDS/model.pth ../nested_unet.pth
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

1.设置环境变量
```
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01
```
atc --framework=5 --model=./nested_unet.onnx --input_format=NCHW --input_shape="actual_input_1:16,3,96,96" --output=nested_unet_bs16 --log=debug --soc_version=Ascend310
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
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 gen_dataset_info.py bin ./prep_dataset ./nested_unet_prep_bin.info 96 96
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息  
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理benchmark工具用户指南 01
### 5.2 离线推理
1.设置环境变量
```
source env.sh
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=nested_unet_bs1.om -input_text_path=./nested_unet_prep_bin.info -input_width=96 -input_height=96 -output_binary=True -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_deviceX(X为对应的device_id)，每个输入对应一个_X.bin文件的输出。

## 6 精度对比

-   **[离线推理IoU精度](#61-离线推理IoU精度)**  
-   **[开源IoU精度](#62-开源IoU精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理IoU精度

后处理统计IoU精度

调用postprocess_nested_unet.py脚本推理结果与语义分割真值进行比对，可以获得IoU精度数据。
```
python3.7 postprocess_nested_unet.py result/dumpOutput_device0/ ./dsb2018_96/masks/0/ 
```
第一个为benchmark输出目录，第二个为真值所在目录。  
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

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 96.455, latency: 1389.25
[data read] throughputRate: 1640.63, moduleLatency: 0.609522
[preprocess] throughputRate: 1633.73, moduleLatency: 0.612097
[infer] throughputRate: 356.828, Interface throughputRate: 428.223, moduleLatency: 2.78137
[post] throughputRate: 356.654, moduleLatency: 2.80384
```
Interface throughputRate: 428.223，428.223x4=1712.892既是batch1 310单卡吞吐率  
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  
```
[e2e] throughputRate: 95.2159, latency: 1407.33
[data read] throughputRate: 12059, moduleLatency: 0.0829254
[preprocess] throughputRate: 5134.3, moduleLatency: 0.194769
[infer] throughputRate: 386.438, Interface throughputRate: 443.958, moduleLatency: 2.5702
[post] throughputRate: 25.8257, moduleLatency: 38.7211
```
Interface throughputRate: 443.958，443.958x4=1775.832既是batch16 310单卡吞吐率  
batch4性能：
```
[e2e] throughputRate: 98.415, latency: 1361.58
[data read] throughputRate: 12401.7, moduleLatency: 0.0806344
[preprocess] throughputRate: 6240.4, moduleLatency: 0.160246
[infer] throughputRate: 450.443, Interface throughputRate: 503.386, moduleLatency: 2.2088
[post] throughputRate: 114.158, moduleLatency: 8.75977
```
batch4 310单卡吞吐率：503.386x4=2013.544fps  
batch8性能：
```
[e2e] throughputRate: 98.2831, latency: 1363.41
[data read] throughputRate: 12465.1, moduleLatency: 0.0802239
[preprocess] throughputRate: 5993.38, moduleLatency: 0.166851
[infer] throughputRate: 439.33, Interface throughputRate: 484.033, moduleLatency: 2.2622
[post] throughputRate: 55.5416, moduleLatency: 18.0045
```
batch8 310单卡吞吐率：484.033x4=1936.132fps  
batch32性能：
```
[e2e] throughputRate: 96.7847, latency: 1384.52
[data read] throughputRate: 14351.5, moduleLatency: 0.0696791
[preprocess] throughputRate: 6115.65, moduleLatency: 0.163515
[infer] throughputRate: 352.844, Interface throughputRate: 386.865, moduleLatency: 2.8028
[post] throughputRate: 13.1131, moduleLatency: 76.2594
```
batch32 310单卡吞吐率：386.865x4=1547.460fps  

 **性能优化：**  
>没有遇到性能不达标的问题，故不需要进行性能优化

