# OSNet Onnx模型端到端推理指导
-   [1 模型概述](#1-模型概述)
	-   [1.1 论文地址](#11-论文地址)
	-   [1.2 代码地址](#12-代码地址)
-   [2 环境说明](#2-环境说明)
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
	-   [6.1 离线推理精度统计](#61-离线推理精度统计)
	-   [6.2 开源精度](#62-开源精度)
	-   [6.3 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[OSNet论文](https://arxiv.org/abs/1905.00953)  
作为一个实例级的识别问题，行人再识别(ReID)依赖于具有识别能力的特征，它不仅能捕获不同的空间尺度，还能封装多个尺度的任意组合。这些同构和异构尺度的特征为全尺度特征。本文设计了一种新颖的深度CNN，称为全尺度网络(OSNet)，用于ReID的全尺度特征学习。这是通过设计一个由多个卷积特征流组成的残差块来实现的，每个残差块检测一定尺度的特征。重要的是，引入了一种新的统一聚合门用输入依赖的每个通道权重进行动态多尺度特征融合。为了有效地学习空间通道相关性，避免过拟合，构建块同时使用点卷积和深度卷积。通过逐层叠加这些块，OSNet非常轻量，可以在现有的ReID基准上从零开始训练。尽管OSNet模型很小，但其在6个Reid数据集上到达了SOTA结果。

### 1.2 代码地址
[OSNet代码](https://github.com/KaiyangZhou/deep-person-reid)  
branch:master   
commit_id:e580b699c34b6f753a9a06223d840317546c98aa   
  
## 2 环境说明
使用CANN版本为CANN:5.1.RC1

深度学习框架与第三方库
```
pytorch == 1.8.1
torchvision == 0.9.1
onnx == 1.7.0
protobu == 3.13.0
onnx-simplifier == 0.3.6
isort == 4.3.21
numpy == 1.18.5
Cython == 0.29.30
h5py == 3.7.0
Pillow == 7.2.0
six == 1.16.0
scipy == 1.7.3
matplotlib == 3.5.2
opencv-python == 4.2.0.34
tb-nightly == 2.10.0a20220527
future == 0.18.2
yacs == 0.1.8
gdown == 4.4.0
flake8 == 4.0.1
yapf == 0.32.0
imageio == 2.19.2
onnxruntime == 1.11.1
onnxoptimizer == 0.2.7
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

#### 1.下载pth权重文件  
 
[OSNet训练pth权重文件(百度网盘下载，提取码：gcfe)](https://pan.baidu.com/s/1Xkwa9TCZss_ygkC8obsEMg)  
osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth

#### 2.下载OSNet源码：
```
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid/
# install dependencies
pip install -r requirements.txt
# install torchreid (don't need to re-build it if you modify the source code)
python setup.py develop
```
#### 3.编写pth2onnx脚本pth2onnx.py，执行pth2onnx脚本，生成onnx模型文件

```
python pth2onnx.py osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth osnet_x1_0.onnx    # 生成onnx模型文件
```
#### 4.对onnx模型进行简化
```
python -m onnxsim osnet_x1_0.onnx osnet_x1_0_bs1_sim.onnx --input-shape 1,3,256,128     # batch_size = 1
python -m onnxsim osnet_x1_0.onnx osnet_x1_0_bs16_sim.onnx --input-shape 16,3,256,128     # batch_size = 16  
```
### 3.2 onnx转om模型

#### 1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
#### 2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)
```
atc --framework=5 --model=./osnet_x1_0_bs1_sim.onnx --input_format=NCHW --input_shape="image:1,3,256,128" --output=osnet_x1_0_bs1 --log=debug --soc_version=Ascend310p     # batch_size = 1
atc --framework=5 --model=./osnet_x1_0_bs16_sim.onnx --input_format=NCHW --input_shape="image:16,3,256,128" --output=osnet_x1_0_bs16 --log=debug --soc_version=Ascend310p      # batch_size = 16
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用Market1501数据集进行测试。Market-1501数据集在清华大学校园中采集，夏天拍摄，在2015年构建并公开。它包括由6个摄像头（其中5个高清摄像头和1个低清摄像头）拍摄的1501个行人的32217张图片。每个行人至少由2个摄像头捕获到，并且在一个摄像头中可能具有多张图像。
训练集bounding_box_train有751人，包含12,936张图像，平均每个人有17.2张训练数据；
测试集bounding_box_test有750人，包含19,732张图像，平均每个人有26.3张测试数据;
查询集query有3368张查询图像。  

Market1501数据集放在/root/datasets/，并将数据集文件夹命名为market1501。

### 4.2 数据集预处理
#### 预处理脚本market1501_torch_preprocess.py，执行预处理脚本，生成数据集预处理后的bin文件
```
# 处理gallery数据集，即bounding_box_test测试集
python market1501_torch_preprocess.py /root/datasets/market1501/bounding_box_test ./gallery_prep_dataset/
# 处理query数据集
python market1501_torch_preprocess.py /root/datasets/market1501/query ./query_prep_dataset/
```
### 4.3 生成数据集信息文件
#### 生成数据集信息文件脚本gen_dataset_info.py   ，执行生成数据集信息脚本，生成gallery和query数据集信息文件
```
python gen_dataset_info.py bin ./gallery_prep_dataset ./gallery_prep_bin.info 128 256
python gen_dataset_info.py bin ./query_prep_dataset ./query_prep_bin.info 128 256
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)
### 5.2 离线推理
```
#对query_prep_bin.info进行处理
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=osnet_x1_0_bs1.om -input_text_path=./query_prep_bin.info -input_width=128 -input_height=256 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_device0，模型只有一个名为feature的输出，每个输入对应的输出对应一个_x.bin文件。

```
#对gallery_prep_bin.info进行处理
./benchmark.x86_64 -model_type=vision -device_id=1 -batch_size=1 -om_path=osnet_x1_0_bs1.om -input_text_path=./gallery_prep_bin.info -input_width=128 -input_height=256 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_device1，模型只有一个名为feature的输出，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计
调用osnet_metrics_market1501_bs1.py脚本，可以获得rank1和mAP数据，结果保存在result_bs1.json中。
```
python osnet_x1_0_metrics_market1501.py result/dumpOutput_device0/ result/dumpOutput_device1/ ./ result_bs1.json
```
第一个为benchmark输出目录，第二个为query数据集配套标签，第三个为gallery数据集配套标签，第四个是生成的文件名。  
查看输出结果：
```
{"title": "Overall statistical evaluation", "value": [{"key": "R1", "value": "0.94299287"}, {"key": "mAP", "value": "0.8257416732159705"}]}
```
### 6.2 开源精度
[OSNet开源代码仓精度](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)
```
模型：osnet_x1_0，R1=94.2%，mAP=82.6%
```
### 6.3 精度对比
将得到的om离线模型推理结果R1、mAP进行比较，与该模型github代码仓上公布的精度对比，R1比代码仓结果略高，mAP下降在1%范围之内，故精度达标。

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
|         | 310      | 310p    | 310p_aoe | T4       | 310p_aoe/310 | 310p_aoe/T4 |
|---------|----------|---------|----------|----------|--------------|-------------|
| bs1     | 1034.756 | 1131.98 | 1597.88  | 426.803  | 1.544        | 3.743834    |
| bs4     | 2570.48  | 3136.56 | 3176     | 940.291  | 1.234        | 3.284802    |
| bs8     | 2306.576 | 3723.82 | 3729.67  | 1407.707 | 1.263        | 1.606165    |
| bs16    | 2483.052 | 3555.71 | 3759.88  | 2261.484 | 3.515        | 1.482646    |
| bs32    | 2457.256 | 3229.45 | 3511.31  | 1906.577 | 1.439        | 1.578819    |
| bs64    | 2386.584 | 2986.71 | 3157.91  | 1303.749 | 1.332        | 2.422257    |
| 最优batch | 2483.052 | 3723.82 | 3759.88  | 2655.71  | 1.499        | 1.6625      |

