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
		- [7.1 310性能数据](#71-310性能数据)
		- [7.2 710性能数据](#72-710性能数据)
		- [7.3 T4性能数据](#73-T4性能数据)
		- [7.4 性能对比](#74-性能对比)



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
CANN 5.1.RC1
pytorch = 1.5.0
torchvision = 0.6.0
onnx = 1.7.0
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

使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN 5.0.1 开发辅助工具指南 (推理) 01]
```
atc --model=./vnet.onnx --framework=5 --output=vnet_bs1 --input_format=NCDHW --input_shape="actual_input_1:1,1,64,80,80" --log=info --soc_version=Ascend310

```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[LUNA16数据集](https://luna16.grand-challenge.org/Download/)的888例CT数据进行肺部区域分割。全部888例CT数据分别存储在subset0.zip~subset9.zip共10个文件中，解压后需要将所有文件移动到vnet.pytorch/luna16/lung_ct_image目录下。另有与CT数据一一对应的分割真值文件存放于seg-lungs-LUNA16.zip文件，将其解压到vnet.pytorch/luna16/seg-lungs-LUNA16目录。
```
cd vnet.pytorch/luna16/lung_ct_image  
wget https://zenodo.org/record/3723295/files/subset0.zip
wget https://zenodo.org/record/4121926/files/subset7.zip
7za x subset0.zip
```
**说明：** 
>   数据集subset0~subset6在3723295链接下载，subset7~subset9在4121926链接下载，解压后lung_ct_image包含888个.raw文件和888个.mhd文件

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
获取推理benchmark工具软件包：解压后获取benchmark工具运行脚本benchmark.{arch}和scripts目录，该目录下包含各种模型处理脚本，包括模型预处理脚本、模型后处理脚本、精度统计脚本等。

获取地址：https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software

Ascend-cann-benchmark_{version}_Linux-{arch}.zip

{version}为软件包的版本号；{arch}为CPU架构，请用户根据实际需要获取对应的软件包。

### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
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
310精度测试结果：
```
Test set: Error: 2497889/439091200 (0.5689%)
```
710精度测试结果：
```
Test set: Error: 2485695/439091200 (0.5661%)
```
经过对batchsize为1/4/8/16/32/64的om测试，精度数据均如上。

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

-   **[310性能数据](#71-310性能数据)**  
-   **[710性能数据](#72-710性能数据)**  
-   **[T4性能数据](#73-T4性能数据)**  
-   **[性能对比](#74-性能对比)**  

### 7.1 310性能数据
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  

batch1：Interface throughputRate: 7.91715
batch4：Interface throughputRate: 8.5008
batch8：Interface throughputRate: 8.00694
batch16：Interface throughputRate: 8.11015
batch32：Interface throughputRate: 7.91441

2.执行parse脚本，计算单卡吞吐率
```
python parse.py result/perf_vision_batchsize_1_device_0.txt
```
batch1_310吞吐率为31.6686fps
batch4_310吞吐率为34.0032fps
batch8_310吞吐率为32.02776fps
batch16_310吞吐率为32.4406fps
batch32_310吞吐率为31.65764fps

### 7.2 710性能数据

batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  

batch1：Interface throughputRate: 65.5303 ,710吞吐率为65.5303fps
batch4：Interface throughputRate: 64.5802 ,710吞吐率为64.5802fps
batch8：Interface throughputRate: 64.3861 ,710吞吐率为64.3861fps
batch16：Interface throughputRate: 63.617 ,710吞吐率为63.617fps
batch32：Interface throughputRate: 59.7592 ,710吞吐率为59.7592fps
batch64：Interface throughputRate: 61.1219 ,710吞吐率为61.1219fps

### 7.3 T4性能数据
在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2  
batch1性能：
```
trtexec --onnx=vnet.onnx --fp16 --shapes=actual_input_1:1x1x64x80x80 --threads
```

batch1 t4单卡吞吐率：1000/(91.687/1)=10.90667fps  
batch4 t4单卡吞吐率：1000/(360.984/4)=11.08082fps
batch8 t4单卡吞吐率：1000/(813.193/8)=9.83776fps
batch16 t4单卡吞吐率：1000/(1563.66/16)=10.41219fps
batch32 t4单卡吞吐率：1000/(5932.02/32)=5.39445fps
batch64 t4单卡吞吐率：1000/(13051.4/64)=4.90369fps

### 7.4 性能对比

310 710 T4性能对比如下(benchmark推理工具)
| batch | 310      | 710     | T4       | 710/310 | 710/T4   |
|-------|----------|---------|----------|---------|----------|
| 1     | 31.6686  | 65.5303 | 10.90667 | 2.06925 | 6.00828  |
| 4     | 34.0032  | 64.5802 | 11.08082 | 1.89924 | 5.82811  |
| 8     | 32.02776 | 64.3861 | 9.83776  | 2.01032 | 6.54479  |
| 16    | 32.4406  | 63.617  | 10.41219 | 1.96103 | 6.10986  |
| 32    | 31.65764 | 59.7592 | 5.39445  | 1.88767 | 11.07790 |
| 64    | -        | 61.1219 | 4.90369  | -       | 12.46447 |
|       |          |         |          |         |          |
| 最优  | 34.0032  | 65.5303 | 11.08082 |         |          |
		
对于所有batchsize，710性能均高于310性能1.2倍，同时710性能均高于T4性能1.6倍，性能达标。  
 **性能优化：**  
>没有遇到性能不达标的问题，故不需要进行性能优化

