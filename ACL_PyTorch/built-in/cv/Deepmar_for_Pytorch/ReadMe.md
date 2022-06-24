# Deepmar Onnx模型端到端推理指导

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
	-   [6.1 离线推理Acc精度统计](#61-离线推理Acc精度统计)
	-   [6.2 开源Acc精度](#62-开源Acc精度)
	-   [6.3 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[deepmar论文](https://ieeexplore.ieee.org/document/7486476)  

### 1.2 代码地址
[deepmar代码](https://github.com/dangweili/pedestrian-attribute-recognition-pytorch.git)  
branch:master  
commit_id:468ae58cf49d09931788f378e4b3d4cc2f171c22

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC1
pytorch == 1.6.0
torchvision == 0.7.0
onnx == 1.7.0
```

### 2.2 python第三方库

```
numpy == 1.18.5
Pillow == 7.2.0
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.准备pth权重文件  
使用训练好的pth权重文件

2.使用开源仓，获取开源命令

```
git clone https://github.com/dangweili/pedestrian-attribute-recognition-pytorch
```
3.将提供的DeepMAR.py替代开源仓“baseline/model”中的同名脚本

4.将提供的export_onnx.py放入开源仓“pedestrian-attribute-recognition-pytorch”目录下。

5.在“pedestrian-attribute-recognition-pytorch”目录下，执行export_onnx.py脚本将.pth.tar文件转换为.onnx文件，执行如下命令。
```
python3 export_onnx.py xxx/checkpoint.pth.tar deepmar_bs1.onnx 1
```
第一个参数为输入权重文件路径，第二个参数为输出onnx文件路径。 第三个参数为batchsize。
运行成功后，在当前目录生成deepmar_bs1.onnx模型文件，然后将deepmar_bs1.onnx	复制到deepmar源码包中。
 **说明：**  
>注意目前ATC支持的onnx算子版本为11


### 3.2 onnx转om模型

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.去除pad算子。

本模型中的pad算子没有用，可以使用remove_pad.py脚本剔除，提升部分性能。

```
执行remove_pad.py脚本。
python3.7 remove_pad.py deepmar_bs1.onnx deepmar_bs1_nopad.pnnx
```


2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01
```
atc \
--model=deepmar_bs1_nopad.onnx \
--framework=5 \
--output=deepmar_bs1 \
--input_format=NCHW \
--input_shape="actual_input_1:1,3,224,224" \
--log=error \
--soc_version=${chip_name}
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用PETA数据集，请用户需自行获取PETA数据集的19000张图片，并取出其中7600张图片作为测试集。可以从Deepmar开源仓中下载数据集。


### 4.2 数据集预处理
数据预处理将原始数据集转换为模型输入二进制格式。通过缩放、均值方差手段归一化，输出为二进制文件。
执行preprocess_deepmar_pytorch.py脚本，保存图像数据到bin文件。
```
python3.7 preprocess_deepmar_pytorch.py /home/HwHiAiUser/dataset/peta/images input_bin image.txt
```
参数说明：
第一个参数为原始数据测试集所在路径，第二个参数为输出的二进制文件（.bin）所在路径，第三个参数为测试集信息。每个图像对应生成一个二进制文件。
执行成功后应在指定目录下产生含有bin文件的文件夹
### 4.3 生成数据集信息文件
使用benchmark推理需要输入二进制数据集的info文件，用于获取数据集。使用get_info.py脚本，输入已经得到的二进制文件，输出生成二进制数据集的info文件，文件记录图片位置和宽高等信息。

```
python3.7 get_info.py ./input_bin deepmar.info 224 224
```
第一个参数为生成的数据集文件格式，第二个参数为预处理后的数据文件的相对路径，第三个参数为生成的数据集文件保存的路径,，最后是图片宽高信息。运行成功后，在当前目录中生成deepmar.info。
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理benchmark工具用户指南 01
### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
增加执行权限
```
chmod u+x benchmark.x86_64
```
执行推理
```
./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=0 -om_path=deepmar_bs1.om -input_width=224 -input_height=224 -input_text_path=deepmar.info -useDvpp=false -output_binary=true
```
输出结果默认保存在当前目录result/dumpOutput_devicex，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理Acc精度统计

后处理统计Acc精度

调用postprocess_deepmar_pytorch.py脚本与数据集标签label.json比对，可以获得	Accuracy数据，结果保存在fusion_result.json中。
```
python3.7 postprocess_deepmar_pytorch.py result/dumpOutput_device0/ label.json
```
第一个为benchmark输出目录，第二个为数据集配套标签
查看输出结果：
```
instance_acc: 0.78965245043699
instance_precision: 0.8823281028182345
instance_recall: 0.8496866930584036
instance_F1: 0.8656998192635741
```
经过对bs1与bs8的om测试，本模型batch1的精度与batch8的精度没有差别，精度数据均如上

### 6.2 开源TopN精度
[开源代码仓精度](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2%2B)

```
Model          Acc     
Deepmar        78.9      
```
### 6.3 精度对比
将得到的om离线模型推理Acc精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device。为快速获取性能数据，也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准。这里给出两种方式，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs8的性能数据为准，对于使用benchmark工具测试的batch4，16，32，64的性能数据在README.md中如下作记录即可。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 202.23, latency: 37580.9
[data read] throughputRate: 244.002, moduleLatency: 4.00367
[preprocess] throughputRate: 217.543, moduleLatency: 4.50029
[infer] throughputRate: 208.333, Interface throughputRate: 326.049, moduleLatency: 4.45783
[post] throughputRate: 208.332, moduleLatency: 4.196
```
Interface throughputRate: 326.049，326.049x4=1304.196既是batch1 310单卡吞吐率

batch8的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_8_device_1.txt：  

```
[e2e] throughputRate: 461.7, latency: 382555
[data read] throughputRate: 570.307, moduleLatency: 1.61574
[preprocess] throughputRate: 497.19, moduleLatency: 2.62255
[infer] throughputRate: 222.175, Interface throughputRate: 508.133, moduleLatency: 3.45377
[post] throughputRate: 27.19833, moduleLatency: 35.976
```
Interface throughputRate: 508.133，491.668x4=2032.532既是batch8 310单卡吞吐率

batch4性能：

```
[e2e] throughputRate: 198.011, latency: 264534
[data read] throughputRate: 236.271, moduleLatency: 5.0436
[preprocess] throughputRate: 123.037, moduleLatency: 5.04955
[infer] throughputRate: 201.874, Interface throughputRate: 435.322, moduleLatency: 4.18727
[post] throughputRate: 50.4682, moduleLatency: 21.0667
```
batch4 310单卡吞吐率：435.322x4=1741.288fps

batch16性能：

```
[e2e] throughputRate: 227.455, latency: 358539
[data read] throughputRate: 276.918, moduleLatency: 7.14704
[preprocess] throughputRate: 246.784, moduleLatency: 7.15391
[infer] throughputRate: 230.734, Interface throughputRate: 542.935, moduleLatency: 3.72351
[post] throughputRate: 14.4666, moduleLatency: 57.2522
```
batch8 310单卡吞吐率：542.935x4=2171.812fps 

batch32性能：

```
[e2e] throughputRate: 227.683, latency: 225547
[data read] throughputRate: 297.234, moduleLatency: 4.25108
[preprocess] throughputRate: 224.935, moduleLatency: 4.2565
[infer] throughputRate: 230.362, Interface throughputRate: 535.007, moduleLatency: 3.51711
[post] throughputRate: 7.95087, moduleLatency: 143.867
```
batch32 310单卡吞吐率：535.007x4=2140.028fps  

batch64性能：

```
[e2e] throughputRate: 203.683, latency: 225547
[data read] throughputRate: 253.234, moduleLatency: 4.25108
[preprocess] throughputRate: 224.935, moduleLatency: 4.2565
[infer] throughputRate: 209.362, Interface throughputRate: 416.653, moduleLatency: 3.51711
[post] throughputRate: 7.95087, moduleLatency: 143.867
```
batch64 310单卡吞吐率：416.653x4=1666.608fps  

**性能优化：**  

>没有遇到性能不达标的问题，故不需要进行性能优化
