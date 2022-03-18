vovnet39 Onnx模型端到端推理指导

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
[vovnet39论文](https://arxiv.org/abs/1904.09730)  

### 1.2 代码地址
[vovnet39代码](https://github.com/AlexanderBurkhart/cnn_train/tree/505637bcd08021e144c94e81401af6bc71fd46c6/VoVNet.pytorch/models_vovnet)  
branch:master  
commit_id:505637bcd08021e144c94e81401af6bc71fd46c6

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.0.1
pytorch == 1.5.0
torchvision == 0.6.0
onnx == 1.7.0
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
[vovnet39预训练pth权重文件](https://www.dropbox.com/s/1lnzsgnixd8gjra/vovnet39_torchvision.pth)  
文件md5sum: 23717a6cadd9729a704f894381444237 

```
http://www.dropbox.com/s/1lnzsgnixd8gjra/vovnet39_torchvision.pth
```
2.vovnet39模型代码在代码仓里

```
github上vovnet39没有安装脚本，在pth2onnx脚本中引用代码仓定义的vovnet39：

git clone https://github.com/AlexanderBurkhart/cnn_train.git
```

3.编写pth2onnx脚本vovnet39_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 vovnet39_pth2onnx.py vovnet39_torchvision.pth vovnet39.onnx
```
 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明  
### 3.2 onnx转om模型

1.设置环境变量
```
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1  开发辅助工具指南 (推理) 01
```
atc --framework=5 --model=./vovnet39.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=vovnet39_bs1 --log=debug --soc_version=Ascend310
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/opt/npu/imagenet/val与/root/datasets/imagenet/val_label.txt。

### 4.2 数据集预处理
1.预处理脚本vovnet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 vovnet_torch_preprocess.py /opt/npu/imagenet/val ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```
python3.7 get_info.py bin ./prep_dataset ./vovnet_prep_bin.info 224 224
```
第一个参数为生成的数据集文件格式，第二个为预处理后的数据文件路径，第三个参数为生成的数据集文件保存的路径，第四个和第五个参数分别为模型输入的宽度和高度
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1  推理benchmark工具用户指南 01
### 5.2 离线推理
1.设置环境变量
```
source env.sh
```
2.执行离线推理

```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=vovnet39_bs1.om -input_text_path=./vovnet_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_devicex，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

后处理统计TopN精度

调用vision_metric_ImageNet.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ /opt/npu/imagenet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：
```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value
": "76.78%"}, {"key": "Top2 accuracy", "value": "86.6%"}, {"key": "Top3 accuracy", "value": "90.23%"}, {"key": "Top4 accuracy", "value": "92.22%"}, {"key": "Top5 accuracy", "value": "93.43%"}]}
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上

### 6.2 开源TopN精度
[开源代码仓精度](https://github.com/AlexanderBurkhart/cnn_train/tree/505637bcd08021e144c94e81401af6bc71fd46c6/VoVNet.pytorch)

```
Model               Acc@1     Acc@5
vovnet39	        76.77     93.43
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device。为快速获取性能数据，也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准。这里给出两种方式，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 190.286, latency: 262762
[data read] throughputRate: 201.42, moduleLatency: 4.96474
[preprocess] throughputRate: 201.059, moduleLatency: 4.97367
[infer] throughputRate: 190.829, Interface throughputRate: 281.608, moduleLatency: 4.54289
[post] throughputRate: 190.829, moduleLatency: 5.2403
```
Interface throughputRate: 281.608，281.608x4=1126.432既是batch1 310单卡吞吐率  
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  

```
[e2e] throughputRate: 132.399, latency: 377646
[data read] throughputRate: 132.922, moduleLatency: 7.5232
[preprocess] throughputRate: 132.751, moduleLatency: 7.53292
[infer] throughputRate: 132.746, Interface throughputRate: 382.566, moduleLatency: 4.0689
[post] throughputRate: 8.2965, moduleLatency: 120.533
```
Interface throughputRate: 382.566，382.566x4=1530.264既是batch16 310单卡吞吐率  
batch4性能：

```
[e2e] throughputRate: 194.167, latency: 257510
[data read] throughputRate: 206.005, moduleLatency: 4.85426
[preprocess] throughputRate: 205.807, moduleLatency: 4.85891
[infer] throughputRate: 194.685, Interface throughputRate: 354.792, moduleLatency: 4.2876
[post] throughputRate: 48.6709, moduleLatency: 20.5462
```
batch4 310单卡吞吐率：354.792x4=1419.168fps  
batch8性能：

```
[e2e] throughputRate: 143.83, latency: 347633
[data read] throughputRate: 144.354, moduleLatency: 6.92743
[preprocess] throughputRate: 144.186, moduleLatency: 6.93549
[infer] throughputRate: 144.246, Interface throughputRate: 364.129, moduleLatency: 4.20264
[post] throughputRate: 18.0306, moduleLatency: 55.4613
```
batch8 310单卡吞吐率：364.129x4=1456.516fps  
batch32性能：

```
[e2e] throughputRate: 122.051, latency: 409664
[data read] throughputRate: 122.581, moduleLatency: 8.15788
[preprocess] throughputRate: 122.445, moduleLatency: 8.16691
[infer] throughputRate: 122.448, Interface throughputRate: 382.396, moduleLatency: 4.07549
[post] throughputRate: 3.82767, moduleLatency: 261.256
```
batch32 310单卡吞吐率：382.396x4=1529.584fps  
  
 **性能优化：**  

>没有遇到性能不达标的问题，故不需要进行性能优化