# PCB Onnx模型端到端推理指导
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
	-   [6.1 开源TopN精度](#62-开源TopN精度)
	-   [6.2 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[PCB论文](https://arxiv.org/pdf/1711.09349.pdf)

分支为 : master

commit ID : e29cf54486427d1423277d4c793e39ac0eeff87c  

### 1.2 代码地址
[PCB开源仓代码](https://github.com/syfafterzy/PCB_RPP_for_reID)

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

### 2.1 深度学习框架
```
python==3.6.7
pytorch==1.8.1
torchvision==0.2.1
```

### 2.2 python第三方库

```
numpy == 1.19.2
scikit-learn == 0.24.1
opencv-python == 4.5.2.54
pillow == 8.2.0
onnx == 1.9.0
pillow == 8.2.0
skl2onnx == 1.8.0
h5py == 3.3.0 
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
[PCB预训练pth权重文件](https://ascend-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/face/PCB/PCB_3_7.pt)  
```
wget https://ascend-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/face/PCB/PCB_3_7.pt

```
 **说明：模型文件名为：PCB_3_7.pt  其md5sum值为：c5bc5ddabcbcc45f127ead797fe8cb35  PCB_3_7.pt**  
>获取的预训练模型放在本仓根目录下

2.编写pth2onnx脚本pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

3.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 pth2onnx.py           #将PCB_3_7.pt模型转为PCB.onnx模型
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
atc --framework=5 --model=PCB.onnx --output=PCB --input_format=NCHW --input_shape="input_1:1,3,384,128" --log=debug --soc_version=Ascend310
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[Market数据集](https://pan.baidu.com/s/1ntIi2Op?_at_=1622802619466)的19732张验证集进行测试。数据集下载后，解压放到./datasets目录下。

### 4.2 数据集预处理
1.预处理脚本PCB_pth_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 PCB_pth_preprocess.py -d market -b 1 --height 384 --width 128 --data-dir ./datasets/Market-1501/ -j 4
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info_Ascend310.sh

2.执行生成数据集信息脚本，生成数据集信息文件
```
sh get_info_Ascend310.sh
```
在get_info_Ascend310.sh文件中调用华为提供的开源工具获取bin文件的路径和尺寸信息，该工具的第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
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
sudo ./benchmark_tools/benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./PCB.om -input_text_path=./gallery_preproc_data_Ascend310.info -input_width=128 -input_height=384 -output_binary=True -useDvpp=False
sudo mv ./result/dumpOutput_device0 ./result/dumpOutput_device0_gallery
sudo mv ./result/perf_vision_batchsize_1_device_0.txt ./result/gallery_perf_vision_batchsize_1_device_0.txt
```
```
sudo ./benchmark_tools/benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./PCB.om -input_text_path=./query_preproc_data_Ascend310.info -input_width=128 -input_height=384 -output_binary=True -useDvpp=False
sudo mv ./result/dumpOutput_device0 ./result/dumpOutput_device0_query
sudo mv ./result/perf_vision_batchsize_1_device_0.txt ./result/query_perf_vision_batchsize_1_device_0.txt
```
输出结果默认保存在当前目录result/dumpOutput_device0下，由于需要通过om模型提取两组特征，因此根据输入图片类型（querry或gallery）分别重命名文件夹。
3.特征图后处理

```
python ./PCB_pth_postprocess.py -q ./result/dumpOutput_device0_query -g ./result/dumpOutput_device0_gallery -d market --data-dir ./datasets/Market-1501/
```
对om模型提取的特征做后处理并统计精度，结果如下：
```
{'title': 'Overall statistical evaluation', 'value': [{'key': 'Number of images', 'value': '15913'}, {'key': 'Number of classes', 'value': '751'}, {'key': 'Top-1 accuracy', 'value': '92.1%'}, {'key': 'Top-5 accuracy', 'value': '96.9%'}, {'key': 'Top-10 accuracy', 'value': '98.1%'}]}
```
## 6 精度对比

-   **[开源TopN精度](#61-开源TopN精度)**  
-   **[精度对比](#62-精度对比)**  

### 6.1 开源TopN精度
```
CMC Scores  market1501
  top-1          92.1%
  top-5          96.9%
  top-10         98.1%
```
### 6.2 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准。  

1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/query_perf_vision_batchsize_1_device_0.txt.txt：
```
-----------------Performance Summary------------------
[e2e] throughputRate: 164.729, latency: 20445.7
[data read] throughputRate: 184.812, moduleLatency: 5.41092
[preprocess] throughputRate: 182.347, moduleLatency: 5.48405
[infer] throughputRate: 175.577, Interface throughputRate: 253.855, moduleLatency: 4.91128
[post] throughputRate: 175.573, moduleLatency: 5.69565
```
Interface throughputRate: 253.855，253.855* 4 = 1015.42既是batch1 310单卡吞吐率


batch4的性能，benchmark工具在整个数据集上推理后生成result/query_perf_vision_batchsize_4_device_0.txt.txt：
```
-----------------Performance Summary------------------
[e2e] throughputRate: 157.081, latency: 21441.2
[data read] throughputRate: 173.63, moduleLatency: 5.75937
[preprocess] throughputRate: 171.283, moduleLatency: 5.83829
[infer] throughputRate: 167.102, Interface throughputRate: 353.841, moduleLatency: 4.32693
[post] throughputRate: 41.7725, moduleLatency: 23.9392
```
Interface throughputRate: 353.841，353.841* 4 = 1415.364既是batch4 310单卡吞吐率


batch8的性能，benchmark工具在整个数据集上推理后生成result/query_perf_vision_batchsize_8_device_0.txt.txt：
```
-----------------Performance Summary------------------
[e2e] throughputRate: 132.514, latency: 25416.1
[data read] throughputRate: 139.993, moduleLatency: 7.14319
[preprocess] throughputRate: 139.054, moduleLatency: 7.19145
[infer] throughputRate: 139.615, Interface throughputRate: 366.98, moduleLatency: 4.21507
[post] throughputRate: 17.4505, moduleLatency: 57.305
```
Interface throughputRate: 366.98，366.98 * 4 = 1467.92既是batch8 310单卡吞吐率

batch16的性能，benchmark工具在整个数据集上推理后生成result/query_perf_vision_batchsize_16_device_0.txt.txt：  
```
-----------------Performance Summary------------------
[e2e] throughputRate: 143.582, latency: 23457
[data read] throughputRate: 150.172, moduleLatency: 6.65904
[preprocess] throughputRate: 148.372, moduleLatency: 6.73981
[infer] throughputRate: 147.201, Interface throughputRate: 362.414, moduleLatency: 4.28791
[post] throughputRate: 9.22071, moduleLatency: 108.452
```
Interface throughputRate: 362.414,362.414 * 4 = 1449.656既是batch16 310单卡吞吐率  


batch32的性能，benchmark工具在整个数据集上推理后生成result/query_perf_vision_batchsize_32_device_0.txt.txt：
```
-----------------Performance Summary------------------
[e2e] throughputRate: 118.266, latency: 28478.2
[data read] throughputRate: 126.885, moduleLatency: 7.88113
[preprocess] throughputRate: 125.442, moduleLatency: 7.97179
[infer] throughputRate: 124.065, Interface throughputRate: 354.632, moduleLatency: 4.30699
[post] throughputRate: 3.90409, moduleLatency: 256.141
```
Interface throughputRate: 354.632，354.632 * 4 = 1418.528既是batch32 310单卡吞吐率

### 7.2 性能优化
原始模型性能不达标原因分析：
根据profiling性能分析的表格，OM模型完成一次离线推理的总耗时中卷积计算（54次）、数据下采样（1次）和数据上采样（1次）这三类操作占总耗时的71%（36%+21%+19%）左右。再往细分，Task ID 95~101总耗时的53.6%，及7%的任务数占了一半以上的耗时。查看对应任务的算子类型，大多为数据转换类：向量尺寸变换和数据类型转换，推测与npu中的算子硬件实现相关。(详见性能分析报告)

原始模型性能与优化后模型性能对比：
batch1：441.128fps(Ascend310) < 1015.42fps(Ascend310)  
batch16：1024.56(Ascend310) < 1449.656fps(Ascend310)  


#### 7.2.1固定模型输入的batch size，并结合onnxsim工具对onnx模型进行优化
优化动机：通过Netron查看onnx的模型结构图发现有一些常量算子可以折叠

优化样例：

    python -m onnxsim --input-shape="16,3,384,128" ./PCB.onnx ./PCB_sim_bs16.onnx

#### 7.42.2.把ReduceL2算子拆分为mul+sum+sqrt算子（无损）
优化动机：Profilingdata可以看到ReduceL2这个算子耗时占比较大，原因是ReduceL2这个算子缺少优化，但是拆分后的算子是经过优化的，且拆分算子后模型的精度保持不变，因此选择拆分ReduceL2算子

优化样例：

    python ../scripts/split_reducelp.py ./PCB_sim_bs16.onnx ./PCB_sim_split_bs16.onnx

#### 7.2.3.atc自动优化选项——autotune
优化动机：atc工具提供的自动优化选项

优化样例：

    atc --framework=5 --model=./PCB_sim_bs4.onnx --output=./PCB_sim_autotune_bs4 --input_format=NCHW --input_shape="input_1:4,3,384,128" --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA"
