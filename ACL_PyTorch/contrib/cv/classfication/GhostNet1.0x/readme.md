# GhostNet1.0x Onnx模型端到端推理指导
-   [1 模型概述](#1-模型概述)
	-   [1.1 论文地址](#11-论文地址)
	-   [1.2 代码地址](#12-代码地址)
-   [2 环境说明](#2-环境说明)
	-   [2.1 深度学习框架](#21-深度学习框架)
	-   [2.2 python第三方库](#22-python第三方库)
-   [3 模型转换](#3-模型转换)
	-   [3.1 pth转onnx模型](#31-pth转onnx模型)
	-   [3.2 onnx转om模型](#32-onnx转om模型)
-   [4 数据集预处理](#4-数据集预处理)
	-   [4.1 数据集获取](#41-数据集获取)
	-   [4.2 数据集预处理](#42-数据集预处理)
	-   [4.3 生成数据集信息文件](#43-生成数据集信息文件)
-   [5 离线推理](#5-离线推理)
	-   [5.1 benchmark工具概述](#51-benchmark工具概述)
	-   [5.2 离线推理](#52-离线推理)
-   [6 精度对比](#6-精度对比)
	-   [6.1 离线推理TopN精度统计](#61-离线推理TopN精度统计)
	-   [6.2 开源TopN精度](#62-开源TopN精度)
	-   [6.3 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[GhostNet论文](https://arxiv.org/abs/1911.11907)  

### 1.2 代码地址
[GhostNet代码](https://github.com/huawei-noah/CV-Backbones/tree/master/ghostnet_pytorch)
branch:master
commit_id:5a06c87a8c659feb2d18d3d4179f344b9defaceb

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
numpy == 1.18.5
Pillow == 7.2.0
opencv-python == 4.5.1.48
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.下载pth权重文件  
[GhostNet预训练pth权重文件](https://github.com/huawei-noah/CV-Backbones/raw/master/ghostnet_pytorch/models/state_dict_73.98.pth)   
文件md5sum:   F7241350B4486BF00ACCBF9C3A192331

```
wget http://github.com/huawei-noah/CV-Backbones/raw/master/ghostnet_pytorch/models/state_dict_73.98.pth
```

2.GhostNet模型代码从如下代码仓中下载
https://github.com/huawei-noah/CV-Backbones/tree/master/ghostnet_pytorch

3.编写pth2onnx脚本ghostnet_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 ghostnet_pth2onnx.py state_dict_73.98.pth ghostnet.onnx
```

 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明

### 3.2 onnx转om模型

1.设置环境变量
```
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)
```
atc --framework=5 --model=./ghostnet.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=ghostnet_bs16 --log=debug --soc_version=Ascend310
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/root/datasets/imagenet/val与/root/datasets/imagenet/val_label.txt。

### 4.2 数据集预处理
1.预处理脚本imagenet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 imagenet_torch_preprocess.py ghostnet /root/datasets/imagenet/val ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 gen_dataset_info.py bin ./prep_dataset ./ghostnet_prep_bin.info 224 224
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)
### 5.2 离线推理
1.设置环境变量
```
source env.sh
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=ghostnet_bs16.om -input_text_path=./ghostnet_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_devicex，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

后处理统计TopN精度

调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ /root/datasets/imagenet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：
```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "73.99%"}, {"key": "Top2 accuracy", "value": "84.0%"}, {"key": "Top3 accuracy", "value": "87.99%"}, {"key": "Top4 accuracy", "value": "90.11%"}, {"key": "Top5 accuracy", "value": "91.46%"}]}
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上

### 6.2 开源TopN精度
[ghostnet代码仓公开模型精度](https://github.com/huawei-noah/CV-Backbones/tree/master/ghostnet_pytorch)
```
Model           Acc@1     Acc@5
ghostnet    	73.98     91.46
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据  
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device。为快速获取性能数据，也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准。这里给出两种方式，模型的测试脚本使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 143.597, latency: 348197
[data read] throughputRate: 153.562, moduleLatency: 6.51203
[preprocess] throughputRate: 152.982, moduleLatency: 6.5367
[infer] throughputRate: 144.973, Interface throughputRate: 196.934, moduleLatency: 6.25397
[post] throughputRate: 144.972, moduleLatency: 6.89786
```
Interface throughputRate: 196.934，196.934x4=787.736既是310单卡吞吐率  
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：
```
[e2e] throughputRate: 111.245, latency: 449458
[data read] throughputRate: 116.28, moduleLatency: 8.59991
[preprocess] throughputRate: 116.21, moduleLatency: 8.60509
[infer] throughputRate: 112.088, Interface throughputRate: 272.671, moduleLatency: 6.92789
[post] throughputRate: 7.00539, moduleLatency: 142.747
```
Interface throughputRate: 272.671，272.671x4=1090.564既是310单卡吞吐率
batch4性能：  
```
[e2e] throughputRate: 152.599, latency: 327655
[data read] throughputRate: 158.671, moduleLatency: 6.30235
[preprocess] throughputRate: 158.538, moduleLatency: 6.30765
[infer] throughputRate: 153.329, Interface throughputRate: 249.801, moduleLatency: 5.61484
[post] throughputRate: 38.332, moduleLatency: 26.0879
```
Interface throughputRate: 249.801，249.801x4=999.204既是310单卡吞吐率  
batch8性能：  
```
[e2e] throughputRate: 164.588, latency: 303789
[data read] throughputRate: 170.932, moduleLatency: 5.85027
[preprocess] throughputRate: 170.715, moduleLatency: 5.8577
[infer] throughputRate: 165.168, Interface throughputRate: 266.985, moduleLatency: 5.32678
[post] throughputRate: 20.6456, moduleLatency: 48.4364
```
Interface throughputRate: 266.985，266.985x4=1067.94既是310单卡吞吐率  
batch32性能：  
```
[e2e] throughputRate: 151.619, latency: 329774
[data read] throughputRate: 152.371, moduleLatency: 6.56292
[preprocess] throughputRate: 152.171, moduleLatency: 6.57154
[infer] throughputRate: 152.123, Interface throughputRate: 283.402, moduleLatency: 5.03802
[post] throughputRate: 4.75527, moduleLatency: 210.293
```
Interface throughputRate: 283.402，283.402x4=1133.608既是310单卡吞吐率  

 **性能优化：**  
从profiling性能数据op_statistic_0_1.csv看出，耗时最多的算子主要是TransData,Conv2D与StridedSliceD，而Conv2D算子不存在性能问题，由于格式转换om模型StridedSliceD前后需要有TransData算子，从op_summary_0_1.csv看出，单个TransData算子aicore耗时不大，单个StridedSliceD算子aicoe耗时也不大，该算子对应的源码也不存在问题，如果优化就需要优化掉过多的TransData算子