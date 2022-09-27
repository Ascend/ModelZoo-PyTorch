
# EDSR ONNX模型端到端推理指导
- [EDSR ONNX模型端到端推理指导](#EDSR-onnx模型端到端推理指导)
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
	- [5 离线推理](#5-离线推理)
		- [5.1 ais-infer工具概述](#51-ais-infer工具概述)
		- [5.2 离线推理](#52-离线推理)
	- [6 精度对比](#6-精度对比)
	- [7 性能对比](#7-性能对比)

## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[EDSR论文](https://arxiv.org/abs/1707.02921)
论文通过提出EDSR模型移除卷积网络中不重要的模块并且扩大模型的规模，使网络的性能得到提升。



### 1.2 代码地址

[EDSR Pytorch实现代码](https://github.com/sanghyun-son/EDSR-PyTorch.git)
```
branch=master 
commit_id=9d3bb0ec620ea2ac1b5e5e7a32b0133fbba66fd2
```



## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架

```
CANN == 5.1.RC2
torch == 1.7.0
torchvision == 0.8.0
onnx == 1.12.0
```



### 2.2 python第三方库

```
opencv-python==4.6.0.66
numpy==1.21.6
Pillow==9.2.0
imageio==2.22.0
matplotlib==3.5.3
tqdm==4.64.1
```

安装必要的依赖，可使用该命令安装

```
pip install -r requirements.txt  
```



## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  


### 3.1 pth转onnx模型

1.获取pth权重文件  

权重文件[链接](https://cv.snu.ac.kr/research/EDSR/model_pytorch.tar)
解压压缩包后获取x2的pt文件，文件名：EDSR_x2.pt



2.获取EDSR源码

```shell
git clone https://github.com/sanghyun-son/EDSR-PyTorch
cd EDSR-PyTorch
git reset --hard 9d3bb0ec620ea2ac1b5e5e7a32b0133fbba66fd2
```



3.运行如下命令，将onnx.diff打包到原仓库中，使得在不影响原仓库功能的前提下实现对官方转换api的支持

```
cd..
patch -p1 < ./edsr.diff
在File to patch一行输入路径：
./EDSR-PyTorch/src/model/__init__.py
```



4.确定onnx输入输出的尺寸 为了增加精度，本指导采用对于不满足尺寸大小要求的图像的右侧和下方填充0的方式来使其输入图像达到尺寸大小要求。因此首先要获得需要的尺寸大小，通过命令行中运行如下脚本

```
python3.7 get_max_size.py --dir /root/datasets/div2k/LR
```
对于div2k数据集中scale为2的缩放，尺寸大小应为1020。



5.使用edsr_pth2onnx.py 脚本将pth转化为onnx

```
python3.7 edsr_pth2onnx.py --pth edsr_x2.pt --onnx edsr_x2.onnx --size 1020
```



### 3.2 onnx转om模型

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```



2.提升OM模型性能(关闭TransposeReshapeFusionPass)。

使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23310P424%7C251366513%7C22892968%7C251168373)，需要指定输出节点以去除无用输出，可以使用netron开源可视化工具查看具体的输出节点名：

使用atc将onnx模型 ${chip_name}可通过npu-smi info指令查看

![输入图片说明](https://images.gitee.com/uploads/images/2022/0704/095450_881600a3_7629432.png "屏幕截图.png")

执行ATC命令，在atc转换命令中加入 --fusion_switch_file=switch.cfg，关闭TransposeReshapeFusionPass。

```shell
atc --model=edsr_x2.onnx --framework=5 --output=edsr_x2 --input_format=NCHW --input_shape="input.1:1,3,1020,1020" --log=debug --soc_version=Ascend${chip_name} --fusion_switch_file=switch.cfg
```

参数说明： \
--model：为ONNX模型文件。 \
--framework：5代表ONNX模型。 \
--output：输出的OM模型。 \
--input_format：输入数据的格式。 \
--input_shape：输入数据的shape。 \
--log：日志级别。 \
--soc_version：处理器型号。 \




## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  
-   **[数据集预处理](#42-数据集预处理)**  

### 4.1 数据集获取
该模型使用[DIV2K官网](https://data.vision.ee.ethz.ch/cvl/DIV2K/)的100张验证集进行测试
其中，低分辨率图像(LR)采用bicubic x2处理(Validation Data Track 1 bicubic downscaling x2 (LR images))，高分辨率图像(HR)采用原图验证集(Validation Data (HR images))。



### 4.2 数据集预处理

执行预处理脚本edsr_preprocess.py，生成数据集预处理后的bin文件
```
python3.7 edsr_preprocess.py -s /root/datasets/div2k/LR -d ./prep_data --save_img
```
预处理脚本会在./prep_data/png/下保存填充为1020x1020的预处理图片，并将bin文件保存至./prep_data/bin/下面。




## 5 离线推理

-   **[ais-infer工具概述](#51-ais-infer工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 ais-infer工具概述

ais-infer工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[链接](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)



### 5.2 离线推理

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.进入ais-infer工具，执行离线推理，执行时使npu-smi info查看设备状态，确保device空闲

```shell
python3.7 ais_infer.py --model ./edsr_x2.om --input ./prep_data/bin --output ./out --batchsize 1
```



## 6 精度对比  

调用edsr_postprocess.py：
```shell
python3.7 edsr_postprocess.py --res ./out --HR /root/datasets/div2k/HR
```

|      | Acc   |
| ---- | ----- |
| 310  | 34.6% |
| 310P | 34.6% |

om模型和官方离线推理精度都为34.6，因此精度达标
没有遇到精度不达标的问题，故不需要进行精度调试



## 7 性能对比

由于T4服务器上的显卡显存有限，性能比较时选择的是size为256的onnx与om模型。

### 7.1 onnx模型转换

size：256的onnx模型生成命令

```
python3.7 edsr_pth2onnx.py --pth edsr_x2.pt --onnx edsr_x2_256.onnx --size 256
```



### 7.2 om模型转换

size：256的om模型生成命令示例如下

```
atc --model=edsr_x2_256.onnx --framework=5 --output=edsr_x2_bs1 --input_format=NCHW --input_shape="input.1:1,3,256,256" --log=debug --soc_version=Ascend${chip_name} --fusion_switch_file=switch.cfg
```



### 7.3 性能测试

npu测试命令示例如下：

```
python3.7 ais_infer.py --model /edsr_x2_256_bs1.om --output ./out --batchsize 1 --loop 20
```

|      | 310 | 310P | T4 | 310P/310 | 310P/T4 |
| ------ | --------- | -------------------------- | ------------ | ----------------------------- | ------ |
| bs1  | 87.543 | 121.239 | 90.272 | 1.384 | 1.343 |
| bs4  | 83.437 | 109.006 | 91.070 | 1.306 | 1.196 |
| bs8  | 83.219 | 113.254 | 92.413 | 1.360 | 1.225 |
| bs16  | 83.504 | 111.150 | 94.146 | 1.331 | 1.180 |
| bs32  | 83.930 | 108.931 | 92.935 | 1.297 | 1.172 |
| bs64  | 83.503 | 107.451 | 91.255 | 1.286 | 1.177 |

