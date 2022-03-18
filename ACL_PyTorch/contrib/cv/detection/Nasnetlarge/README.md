# Nasnetlarge Onnx模型端到端推理指导
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
	-   [6.1 离线推理精度统计](#61-离线推理精度统计)
	-   [6.2 开源精度](#62-开源精度)
	-   [6.3 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)
	-   [7.2 基准性能数据](#72-基准性能数据)
	-   [7.3 性能对比](#73-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[论文地址](https://arxiv.org/abs/1707.07012)  

### 1.2 代码地址
[代码地址](https://github.com/Cadene/pretrained-models.pytorch#nasnet)  
branch:master    
commit id：b8134c79b34d8baf88fe0815ce6776f28f54dbfe
## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
python3.7.5
CANN 5.0.1

pytorch >= 1.8.0
torchvision >= 0.9.0
onnx >= 1.8.0
```
### 2.2 python第三方库

```
numpy == 1.20.3
munch == 2.5.1.dev12
tqdm == 4.36.1
scipy == 1.3.1
onnx-simplifier == 0.3.5 
skl2onnx
tqdm
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
[nasnetlarge预训练pth权重文件](http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth) 
md5sum：78a73e51ee50997294f1f35a34b4de66   
```
wget http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth
```
安装onnx_tools

git clone https://gitee.com/zheng-wengang1/onnx_tools.git test/onnx_tools
```
2.编写pth2onnx脚本nasnetlarge_pth2onnx.py

 **说明：**  
>目前ATC支持的onnx算子版本为11

3.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 nasnetlarge_pth2onnx.py nasnetalarge-a1897284.pth nasnetlarge.onnx
```
4.使用onnxsim，生成onnx_sim模型文件
```
python3.7 -m onnxsim --input-shape="1,3,331,331" nasnetlarge.onnx nasnetlarge_sim.onnx
```
5.算子融合优化
```
python3.7 merge_sliced.py nasnetlarge_sim.onnx nasnetlarge_sim_merge.onnx
```

### 3.2 onnx转om模型

1.设置环境变量
```
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)
```
atc --framework=5 --model=nasnetlarge_sim_merge.onnx --input_format=NCHW --input_shape="image:1,3,331,331" --output=nasnetlarge_sim_bs1 --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA" 
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  
-   **[数据集预处理](#42-数据集预处理)**  
-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
对于图像分类任务，该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/opt/npu/imagenet/val与/opt/npu/imagenet/val_label.txt

### 4.2 数据集预处理
1.预处理脚本preprocess_img.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 preprocess_img.py /opt/npu/imagenet/val ./prep_dataset 
```
第一个参数为验证集路径，第二个参数为预处理后生成的二进制文件的存储路径

### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 gen_dataset_info.py bin ./prep_dataset ./nasnetlarge_prep_bin.info 331 331
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
 ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=nasnetlarge_sim_bs1.om -input_text_path=./nasnetlarge_prep_bin.info -input_width=331 -input_height=331 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_devicex，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据。
```
python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ /opt/npu/imagenet/val_label.txt ./ result.json
```
第一个参数为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名，其中存有推理的Top5精度。
对batch1和batch16的模型分别调用benchmark进行推理，并统计其Top5的精度。查看其输出结果：
```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1001"}, {"key": "Top1 accuracy", "value": "82.53%"}, {"key": "Top2 accuracy", "value": "91.12%"}, {"key": "Top3 accuracy", "value": "93.91%"}, {"key": "Top4 accuracy", "value": "95.2%"}, {"key": "Top5 accuracy", "value": "95.99%"}]}
```
经过对bs1与bs16的om测试，本模型batch1与batch16的精度没有差别，精度数据均如上。

### 6.2 开源TopN精度
[github开源代码仓精度](https://github.com/Cadene/pretrained-models.pytorch/blob/master/README.md)
```
Model               Acc@1     Acc@5
NASNet-Large		82.566	  96.086
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，Top1与Top5精度均达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[基准性能数据](#72-基准性能数据)**  
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时会统计性能数据，存储于result/perf_vision_batchsize_bs_device_0.txt中。但是推理整个数据集较慢，如此测性能时需要确保benchmark独占device，使用npu-smi info可以查看device是否空闲。
除此之外，也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。
模型的性能以使用benchmark工具推理得到bs1与bs16的性能数据为准；对于使用benchmark工具测试的batch4，8，32的性能数据仅在README.md中作如下记录。  
1.benchmark工具推理获得性能数据  
使用benchmark工具的纯推理功能测试模型的推理性能，命令如下：
```
./benchmark.x86_64 -round=20 -om_path=nasnetlarge_sim_bsx.om -device_id=0 -batch_size=x
x为batch size,取值为1，4，8，16，32。
```
benchmark工具进行纯推理后测得的性能数据存储于result/PureInfer_perf_of_mobilenet-v1_bsx_in_device_0.txt，其中x为模型的batch_size。

batch1性能：  
```
-----------------PureInfer Performance Summary-----------------
[INFO] ave_throughputRate = 7.18646samples/s, ave_latency = 139.358ms
```
ave_throughputRate是npu单核的平均吞吐率，乘以4即为310单卡的吞吐率。即：batch1的310单卡吞吐率为7.186x4=28.744fps

batch16性能：
```
-----------------PureInfer Performance Summary-----------------
[INFO] ave_throughputRate: 7.46549samples/s, ave_latency: 134.029ms
```
batch16 310单卡吞吐率 ：7.465x4=29.86fps

batch4性能：
```
-----------------PureInfer Performance Summary-----------------
[INFO] ave_throughputRate: 7.30466samples/s, ave_latency: 137.271ms
```
batch4 310单卡吞吐率：7.304x4=29.216fps  

batch8性能：
```
-----------------PureInfer Performance Summary-----------------
[INFO] ave_throughputRate: 7.33096samples/s, ave_latency: 136.514ms
```
batch8 310单卡吞吐率：7.33x4=29.32fps  

batch32性能：
```
-----------------PureInfer Performance Summary-----------------
[INFO] ave_throughputRate: 7.33135samples/s, ave_latency: 136.423ms
```
batch32 310单卡吞吐率：7.331x4=29.324fps  

### 7.2 基准性能数据
在装有基准卡的服务器上测试gpu性能，测试前请使用nvidia-smi命令确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2  

batch1性能：
```
 ./trtexec --onnx=nasnetlarge_sim.onnx --fp16 --shapes=image:1x3x331x331 --threads
```
gpu 基准是4个device并行执行的结果，mean是对batch_size个输入做推理的平均时延，故吞吐率为1000/(mean/batch_size) fps。
```
[06/23/2021-14:18:36] [I] min: 12.3789 ms
[06/23/2021-14:18:36] [I] max: 17.5124 ms
[06/23/2021-14:18:36] [I] mean: 12.8522 ms
[06/23/2021-14:18:36] [I] median: 12.7339 ms

```
batch1 基准单卡吞吐率：1000/(12.73/1)=78.55fps  

batch16性能：
```
 ./trtexec --onnx=nasnetlarge_sim.onnx --fp16 --shapes=image:16x3x331x331 --threads
```
```
[06/23/2021-16:00:32] [I] GPU Compute
[06/23/2021-16:00:32] [I] min: 144.1865ms
[06/23/2021-16:00:32] [I] max: 144.4489 ms
[06/23/2021-16:00:32] [I] mean: 144.8318 ms
[06/23/2021-16:00:32] [I] median: 145.0719 ms

```
batch16 基准单卡吞吐率：1000/(12.242/16)=110.29fps  

batch4性能：
```
[06/23/2021-16:18:13] [I] GPU Compute
[06/23/2021-16:18:13] [I] min: 43.0629 ms
[06/23/2021-16:18:13] [I] max: 44.1789 ms
[06/23/2021-16:18:13] [I] mean: 43.8496 ms
[06/23/2021-16:18:13] [I] median: 43.8327 ms

```
batch4 基准单卡吞吐率：1000/(43.83/4)=91.26fps  

batch8性能：
```
[06/24/2021-02:52:28] [I] GPU Compute
[06/24/2021-02:52:28] [I] min: 80.6491 ms
[06/24/2021-02:52:28] [I] max: 81.8325 ms
[06/24/2021-02:52:28] [I] mean: 80.9361 ms
[06/24/2021-02:52:28] [I] median: 81.1688 ms

```
batch8 基准单卡吞吐率：1000/(81.1688/8)=98.56fps  

batch32性能：
```
[06/24/2021-04:27:40] [I] GPU Compute
[06/24/2021-04:27:40] [I] min: 262.1743 ms
[06/24/2021-04:27:40] [I] max: 263.0174 ms
[06/24/2021-04:27:40] [I] mean: 262.7492 ms
[06/24/2021-04:27:40] [I] median: 262.8768 ms

```
batch32 基准单卡吞吐率：1000/(262.8768/32)=121.73fps  

### 7.3 性能对比
batch1：28.744 < 78.55  
batch16：29.86 < 110.29  
对于batch1与batch16的模型，310的单卡吞吐率均小于基准单卡的吞吐率，性能未达标。
 
转换后的om模型在性能上并未达标，经过profiling工具的分析，拉低模型推理性能的主要原因如下：StridedSliceD和TransData部分耗时过长。经过simplyfier优化、添加Transdata白名单优化、autotune优化、merge_slice优化后性能有提升。



