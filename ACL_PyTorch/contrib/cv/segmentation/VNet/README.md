# VNet Onnx模型端到端推理指导
- [VNet Onnx模型端到端推理指导](#vnet-onnx模型端到端推理指导)
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
		- [6.1 离线推理精度](#61-离线推理精度)
		- [6.2 开源精度](#62-开源精度)
		- [6.3 精度对比](#63-精度对比)
	- [7 性能对比](#7-性能对比)
		- [7.1 npu性能数据](#71-npu性能数据)
		- [7.2 T4性能数据](#72-t4性能数据)
		- [7.3 性能对比](#73-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[VNet论文](https://arxiv.org/abs/1606.04797)  

### 1.2 代码地址
[VNet代码](https://github.com/mattmacy/vnet.pytorch)  
branch:master  
commit_id:a00c8ea16bcaea2bddf73b2bf506796f70077687  
备注：commit_id是指基于该次提交时的模型代码做推理，通常选择稳定版本的最后一次提交，或代码仓最新的一次提交  

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.0.3.alpha002 
pytorch >= 1.5.0
torchvision >= 0.6.0
onnx >= 1.7.0
```

### 2.2 python第三方库

```
numpy == 1.20.3
opencv-python == 4.5.2.54
SimpleITK == 2.1.0
tqdm == 4.61.1
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.VNet模型代码下载
```
git clone https://github.com/mattmacy/vnet.pytorch
cd vnet.pytorch
git checkout a00c8ea16bcaea2bddf73b2bf506796f70077687
```
2.对原代码进行修改，以满足数据集预处理及模型转换等功能。
```
patch -p1 < ../vnet.patch
cd ..
```

3.获取权重文件vnet_model_best.pth.tar

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 vnet_pth2onnx.py vnet_model_best.pth.tar vnet.onnx
```

### 3.2 onnx转om模型

1.设置环境变量
```
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN 5.0.1 开发辅助工具指南 (推理) 01]
```
atc --model=./vnet.onnx --framework=5 --output=vnet_bs1 --input_format=NCDHW --input_shape="actual_input_1:1,1,64,80,80" --log=info --soc_version=Ascend310

```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[LUNA16数据集](https://luna16.grand-challenge.org/Download/)的888例CT数据进行肺部区域分割。全部888例CT数据分别存储在subset0.zip~subset9.zip共10个文件中，解压后需要将所有文件移动到vnet.pytorch/luna16/lung_ct_image目录下。另有与CT数据一一对应的分割真值文件存放于seg-lungs-LUNA16.zip文件，将其解压到vnet.pytorch/luna16/seg-lungs-LUNA16目录。

### 4.2 数据集预处理
1.执行原代码仓提供的数据集预处理脚本。
```
cd vnet.pytorch  
python normalize_dataset.py ./luna16 2.5 128 160 160  
cd ..
```

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 vnet_preprocess.py ./vnet.pytorch/luna16 ./prep_bin ./vnet.pytorch/test_uids.txt
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 gen_dataset_info.py bin ./prep_bin ./vnet_prep_bin.info 80 80
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息  
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN 5.0.1 推理benchmark工具用户指南 01]
### 5.2 离线推理
1.设置环境变量
```
source env.sh
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=vnet_bs1.om -input_text_path=./vnet_prep_bin.info -input_width=80 -input_height=80 -output_binary=True -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_deviceX(X为对应的device_id)，每个输入对应一个_X.bin文件的输出。

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度

后处理统计精度

调用vnet_postprocess.py脚本将推理结果与语义分割真值进行比对，可以获得精度数据。
```
python3.7 vnet_postprocess.py result/dumpOutput_device0 ./vnet.pytorch/luna16/normalized_lung_mask ./vnet.pytorch/test_uids.txt
```
第一个为benchmark输出目录，第二个为真值所在目录，第三个为测试集样本的序列号。  
查看输出结果：
```
Error rate: 2479051/439091200 (0.5646%)
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上。

### 6.2 开源精度
[原代码仓公布精度](https://github.com/mattmacy/vnet.pytorch/blob/master/README.md)
```
Model   Error rate 
VNet    0.355% 
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
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 5.70609, latency: 187869
[data read] throughputRate: 225.606, moduleLatency: 4.43251
[preprocess] throughputRate: 53.7844, moduleLatency: 18.5928
[inference] throughputRate: 5.75202, Interface throughputRate: 6.10496, moduleLatency: 173.468
[postprocess] throughputRate: 5.75712, moduleLatency: 173.698
```
Interface throughputRate: 6.10496，6.10496x4=24.41984既是batch1 310单卡吞吐率  
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  
```
[e2e] throughputRate: 6.24092, latency: 171769
[data read] throughputRate: 377.232, moduleLatency: 2.65089
[preprocess] throughputRate: 61.2764, moduleLatency: 16.3195
[inference] throughputRate: 6.2793, Interface throughputRate: 6.49396, moduleLatency: 159.033
[postprocess] throughputRate: 0.398022, moduleLatency: 2512.42
```
Interface throughputRate: 6.49396，6.49396x4=25.97584既是batch16 310单卡吞吐率  
batch4性能：
```
[e2e] throughputRate: 6.38643, latency: 167856
[data read] throughputRate: 220.829, moduleLatency: 4.52839
[preprocess] throughputRate: 59.272, moduleLatency: 16.8714
[inference] throughputRate: 6.42624, Interface throughputRate: 6.67466, moduleLatency: 155.341
[postprocess] throughputRate: 1.61227, moduleLatency: 620.245
```
batch4 310单卡吞吐率：6.67466x4=26.69864fps  
batch8性能：
```
[e2e] throughputRate: 6.17056, latency: 173728
[data read] throughputRate: 216.73, moduleLatency: 4.61403
[preprocess] throughputRate: 57.3928, moduleLatency: 17.4238
[inference] throughputRate: 6.20835, Interface throughputRate: 6.41992, moduleLatency: 160.848
[postprocess] throughputRate: 0.781576, moduleLatency: 1279.47
```
batch8 310单卡吞吐率：6.41992x4=25.67968fps  
batch32性能：
```
[e2e] throughputRate: 6.09413, latency: 175907
[data read] throughputRate: 183.187, moduleLatency: 5.45889
[preprocess] throughputRate: 49.9254, moduleLatency: 20.0299
[inference] throughputRate: 6.15986, Interface throughputRate: 6.35151, moduleLatency: 162.051
[postprocess] throughputRate: 0.200903, moduleLatency: 4977.52
```
batch32 310单卡吞吐率：6.35151x4=25.40604fps  

### 7.2 T4性能数据
在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2  
batch1性能：
```
trtexec --onnx=vnet.onnx --fp16 --shapes=actual_input_1:1x1x64x80x80 --threads
```
gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch
```
[09/17/2021-15:39:40] [I] GPU Compute
[09/17/2021-15:39:40] [I] min: 92.4146 ms
[09/17/2021-15:39:40] [I] max: 103.909 ms
[09/17/2021-15:39:40] [I] mean: 97.0678 ms
[09/17/2021-15:39:40] [I] median: 96.9087 ms
[09/17/2021-15:39:40] [I] percentile: 103.909 ms at 99%
[09/17/2021-15:39:40] [I] total compute time: 3.20324 s
```
batch1 t4单卡吞吐率：1000/(96.9087/1)=10.31899fps  

batch16性能：
```
trtexec --onnx=nested_unet.onnx --fp16 --shapes=actual_input_1:16x3x96x96 --threads
```
```
[09/17/2021-16:11:37] [I] GPU Compute
[09/17/2021-16:11:37] [I] min: 1574.28 ms
[09/17/2021-16:11:37] [I] max: 1576.2 ms
[09/17/2021-16:11:37] [I] mean: 1575.22 ms
[09/17/2021-16:11:37] [I] median: 1574.94 ms
[09/17/2021-16:11:37] [I] percentile: 1576.2 ms at 99%
[09/17/2021-16:11:37] [I] total compute time: 15.7522 s
```
batch16 t4单卡吞吐率：1000/(1575.22/16)=10.15731fps  

batch4性能：
```
[09/17/2021-15:44:51] [I] GPU Compute
[09/17/2021-15:44:51] [I] min: 361.722 ms
[09/17/2021-15:44:51] [I] max: 375.435 ms
[09/17/2021-15:44:51] [I] mean: 365.263 ms
[09/17/2021-15:44:51] [I] median: 363.615 ms
[09/17/2021-15:44:51] [I] percentile: 375.435 ms at 99%
[09/17/2021-15:44:51] [I] total compute time: 3.65263 s
```
batch4 t4单卡吞吐率：1000/(365.263/4)=10.95101fps  

batch8性能：
```
[09/17/2021-15:52:50] [I] GPU Compute
[09/17/2021-15:52:50] [I] min: 796.131 ms
[09/17/2021-15:52:50] [I] max: 802.935 ms
[09/17/2021-15:52:50] [I] mean: 798.473 ms
[09/17/2021-15:52:50] [I] median: 798.262 ms
[09/17/2021-15:52:50] [I] percentile: 802.935 ms at 99%
[09/17/2021-15:52:50] [I] total compute time: 7.98473 s
```
batch8 t4单卡吞吐率：1000/(798.473/8)=10.01912fps  

batch32性能：
```
[09/17/2021-16:29:35] [I] GPU Compute
[09/17/2021-16:29:35] [I] min: 3382.94 ms
[09/17/2021-16:29:35] [I] max: 3395.54 ms
[09/17/2021-16:29:35] [I] mean: 3389.83 ms
[09/17/2021-16:29:35] [I] median: 3390.36 ms
[09/17/2021-16:29:35] [I] percentile: 3395.54 ms at 99%
[09/17/2021-16:29:35] [I] total compute time: 33.8983 s
```
batch32 t4单卡吞吐率：1000/(3389.83/32)=9.44fps  

### 7.3 性能对比
batch1：6.10496x4 > 1000x1/(96.9087/1)  
batch16：6.49396x4 > 1000x1/(1575.22/16)  
310单个device的吞吐率乘4即单卡吞吐率比T4单卡的吞吐率大，故310性能高于T4性能，性能达标。  
对于batch1与batch16，310性能均高于T4性能1.2倍，该模型放在ACL_PyTorch/Benchmark/cv/segmentation目录下。  
 **性能优化：**  
>没有遇到性能不达标的问题，故不需要进行性能优化

