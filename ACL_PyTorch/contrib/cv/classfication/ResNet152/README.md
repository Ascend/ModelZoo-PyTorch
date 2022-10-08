# ResNet152 Onnx模型端到端推理指导
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
	-   [7.2 T4性能数据](#72-T4性能数据)
	-   [7.3 性能对比](#73-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[ResNet152论文](https://arxiv.org/pdf/1512.03385.pdf)  

### 1.2 代码地址
[ResNet152代码](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)  
branch:master
commit_id:02e6da5189b22870c549470485d68fff23d511bf
          

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
opencv-python == 4.5.1.52
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
[ResNet152预训练pth权重文件](https://download.pytorch.org/models/resnet152-b121ed2d.pth)  
```
wget https://download.pytorch.org/models/resnet152-b121ed2d.pth
```
文件MD5sum：d3ddb494358a7e95e49187829ec97395

2.ResNet152模型代码在torchvision里，安装torchvision，arm下需源码安装，参考torchvision官网，若安装过程报错请百度解决
```
git clone https://github.com/pytorch/vision
cd vision
python3.7 setup.py install
cd ..
```
3.编写pth2onnx脚本resnet152_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 resnet152_pth2onnx.py ./resnet152-f37072fd.pth resnet152.onnx
```

 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明

### 3.2 onnx转om模型

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)
```
atc --framework=5 --model=./resnet152.onnx --output=resnet152_bs32 --input_format=NCHW --input_shape="image:32,3,224,224" --log=debug --soc_version=Ascend310

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
python3.7 imagenet_torch_preprocess.py resnet /root/datasets/imagenet/val ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 gen_dataset_info.py bin ./prep_dataset ./resnet152_prep_bin.info 224 224
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
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=resnet152_bs1.om -input_text_path=./resnet152_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=resnet152_bs16.om -input_text_path=./resnet152_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=4 -om_path=resnet152_bs4.om -input_text_path=./resnet152_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

```
输出结果默认保存在当前目录result/dumpOutput_device{0}，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

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
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "78.31%"}, {"key": "Top2 accuracy", "value": "87.83%"}, {"key": "Top3 accuracy", "value": "91.25%"}, {"key": "Top4 accuracy", "value": "92.97%"}, {"key": "Top5 accuracy", "value": "94.05%"}]}
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上

### 6.2 开源TopN精度
[torchvision官网精度](https://pytorch.org/vision/stable/models.html)
```
Model        Acc@1     Acc@5
resnet152    78.312	   94.046
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[T4性能数据](#72-T4性能数据)**  
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 126.32, latency: 395819
[data read] throughputRate: 134.323, moduleLatency: 7.44476
[preprocess] throughputRate: 133.845, moduleLatency: 7.47134
[infer] throughputRate: 127.16, Interface throughputRate: 180.131, moduleLatency: 6.90735
[post] throughputRate: 127.159, moduleLatency: 7.86415
```
Interface throughputRate: 180.131，180.131x4=720.524既是batch1 310单卡吞吐率  

batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  
```
[e2e] throughputRate: 156.123, latency: 320261
[data read] throughputRate: 165.096, moduleLatency: 6.05708
[preprocess] throughputRate: 164.628, moduleLatency: 6.07431
[infer] throughputRate: 156.862, Interface throughputRate: 291.04, moduleLatency: 5.1523
[post] throughputRate: 9.80371, moduleLatency: 102.002
```
Interface throughputRate: 291.04，291.04x4=1,164.16既是batch16 310单卡吞吐率  


./benchmark.x86_64 -batch_size=4 -om_path=./model_rectify_random.onnx.om -round=50 -device_id=0
batch4的性能，benchmark工具纯推理后生成result/PureInfer_perf_of_resnet152_bs4_in_device_0.txt：  
```

ave_throughputRate = 244.742samples/s, ave_latency = 4.15733ms

```
Interface throughputRate: 244.742，244.742x4=978.968既是batch4 310单卡吞吐率 

batch8的性能，benchmark工具进行纯推理生成result/PureInfer_perf_of_resnet152_bs8_in_device_0.txt：  
```
ave_throughputRate = 270.962samples/s, ave_latency = 3.75835ms

```
Interface throughputRate: 270.962，270.962x4=1,083.848既是batch8 310单卡吞吐率   

batch32的性能，benchmark工具纯推理后生成result/PureInfer_perf_of_resnet152_bs32_in_device_0.txt：  
```
ave_throughputRate = 270.402samples/s, ave_latency = 3.71981ms

```
Interface throughputRate: 270.402，270.402x4=1,081.608既是batch32 310单卡吞吐率  

### 7.2 T4性能数据
在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2  
batch1性能：
```
trtexec --onnx=resnet152.onnx --fp16 --shapes=image:1x3x224x224 --threads
```
gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch
```
[06/11/2021-02:43:51] [I] GPU Compute
[06/11/2021-02:43:51] [I] min: 2.9082 ms
[06/11/2021-02:43:51] [I] max: 6.05182 ms
[06/11/2021-02:43:51] [I] mean: 3.00433 ms
[06/11/2021-02:43:51] [I] median: 2.97778 ms
[06/11/2021-02:43:51] [I] percentile: 3.16479 ms at 99%
[06/11/2021-02:43:51] [I] total compute time: 3.00133 s
```
batch1 t4单卡吞吐率：1000/(3.00433/1)=332.8529156251144fps 
```
 

batch16性能：
```
trtexec --onnx=resnet152.onnx --fp16 --shapes=image:16x3x224x224 --threads
```
[06/11/2021-02:50:44] [I] GPU Compute
[06/11/2021-02:50:44] [I] min: 19.9592 ms
[06/11/2021-02:50:44] [I] max: 22.4021 ms
[06/11/2021-02:50:44] [I] mean: 21.0969 ms
[06/11/2021-02:50:44] [I] median: 20.9503 ms
[06/11/2021-02:50:44] [I] percentile: 22.3171 ms at 99%
[06/11/2021-02:50:44] [I] total compute time: 3.03795 s
```
batch16 t4单卡吞吐率：1000/(21.0969/16)=758.4052633325275fps  


batch4性能：
```
[06/11/2021-08:01:43] [I] GPU Compute
[06/11/2021-08:01:43] [I] min: 6.2175 ms
[06/11/2021-08:01:43] [I] max: 12.7552 ms
[06/11/2021-08:01:43] [I] mean: 6.57256 ms
[06/11/2021-08:01:43] [I] median: 6.47629 ms
[06/11/2021-08:01:43] [I] percentile: 6.98999 ms at 99%
[06/11/2021-08:01:43] [I] total compute time: 3.01023 s

```
batch4 t4单卡吞吐率：1000/(6.57256/4)=608.590868702606fps 



batch8性能：
```
[06/11/2021-08:03:59] [I] GPU Compute
[06/11/2021-08:03:59] [I] min: 10.8062 ms
[06/11/2021-08:03:59] [I] max: 12.296 ms
[06/11/2021-08:03:59] [I] mean: 11.3813 ms
[06/11/2021-08:03:59] [I] median: 11.2798 ms
[06/11/2021-08:03:59] [I] percentile: 12.2863 ms at 99%
[06/11/2021-08:03:59] [I] total compute time: 3.02744 s

```
batch8 t4单卡吞吐率：1000/(11.3813/8)=702.9074007362955fps 



batch32性能：
```
[06/11/2021-08:06:39] [I] GPU Compute
[06/11/2021-08:06:39] [I] min: 39.5345 ms
[06/11/2021-08:06:39] [I] max: 52.6029 ms
[06/11/2021-08:06:39] [I] mean: 44.5667 ms
[06/11/2021-08:06:39] [I] median: 43.1216 ms
[06/11/2021-08:06:39] [I] percentile: 52.6029 ms at 99%
[06/11/2021-08:06:39] [I] total compute time: 3.0751 s

```
batch32 t4单卡吞吐率：1000/(44.5667/32)=718.0248930255101fps 

```

### 7.3 性能对比
batch1：180.131x4 > 1000/(3.00433/1)  
batch16：291.04x4 > 1000/(21.0969/16)  
310单个device的吞吐率乘4即单卡吞吐率比T4单卡的吞吐率大，故310性能高于T4性能，性能达标。  
对于batch1的310性能高于T4性能2.16倍，batch16的310性能高于T4性能1.535倍，该模型放在Benchmark/cv/classification目录下。  
 **性能优化：**  
>没有遇到性能不达标的问题，故不需要进行性能优化

