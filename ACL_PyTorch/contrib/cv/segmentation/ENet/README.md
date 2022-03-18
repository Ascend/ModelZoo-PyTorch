# ENet Onnx模型端到端推理指导
- [ENet Onnx模型端到端推理指导](#ENet onnx模型端到端推理指导)
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
		- [6.1 离线推理IoU精度](#61-离线推理iou精度)
		- [6.2 精度对比](#62-精度对比)
	- [7 性能对比](#7-性能对比)
		- [7.1 npu性能数据](#71-npu性能数据)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[ENet论文](https://arxiv.org/pdf/1606.02147.pdf)  

### 1.2 代码地址
[ENet代码](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)  
branch:master  
commit_id: **5843f75215dadc5d734155a238b425a753a665d9**  
备注：commit_id是指基于该次提交时的模型代码做推理，通常选择稳定版本的最后一次提交，或代码仓最新的一次提交  

上述开源代码仓库没有给出训练好的模型权重文件，因此使用910训练好的pth权重文件来做端到端推理，该权重文件的精度是**54.627%**。

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
numpy == 1.20.3
Pillow == 8.4.0
opencv-python == 4.5.2.54
albumentations == 0.4.5
densetorch == 0.0.2
```

**说明：** 

>   X86架构：pytorch和torchvision可以通过官方下载whl包安装，其他可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和opencv可以通过github下载源码编译安装，其他可以通过pip3.7 install 包名 安装
>
>   以上为多数网络需要安装的软件与推荐的版本，根据实际情况安装。如果python脚本运行过程中import 模块失败，安装相应模块即可，如果报错是缺少动态库，网上搜索报错信息找到相应安装包，执行apt-get install 包名安装即可

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.编写pth2onnx脚本RefineNet_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

2.执行pth2onnx脚本，生成onnx模型文件

```bash
python3.7 ENet_pth2onnx.py --input-file models/enet_citys.pth --output-file models/enet_citys_910_bs1.onnx --batch-size 1
```

### 3.2 onnx转om模型

1.设置环境变量

```
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01

```BASH
atc --framework=5 --model=models/enet_citys_910_bs1_sim.onnx --output=models/enet_citys_910_bs1 --input_format=NCHW --input_shape="image:1,3,480,480" --log=info --soc_version=Ascend310
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用Cityscapes数据集作为训练集，其下的val中的500张图片作为验证集。推理部分只需要用到这500张验证图片，验证集输入图片存放在`citys/leftImg8bit/val`，验证集target存放在`/citys/gtFine/val`。

下载Cityscapes数据集后，把文件夹解压放在`/opt/npu`下。

### 4.2 数据集预处理
1.参考开源代码仓库对验证集所做的预处理，编写预处理脚本。

2.执行预处理脚本，生成数据集预处理后的bin文件

```bash
python3.7 ENet_preprocess.py --src-path=$datasets_path --save_path ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```bash
python3.7 get_info.py bin ./prep_dataset ./enet_prep_bin.info 480 480
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息  
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理benchmark工具用户指南 01
### 5.2 离线推理
1.设置环境变量
```
source env.sh
```
2.执行离线推理
```bash
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=models/enet_citys_910_bs1.om -input_text_path=./enet_prep_bin.info -input_width=480 -input_height=480 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_deviceX(X为对应的device_id)，每个输入对应一个_X.bin文件的输出。

## 6 精度对比

-   **[离线推理MIoU精度](#61-离线推理IoU精度)**    
-   **[精度对比](#62-精度对比)**  

### 6.1 离线推理MIoU精度

后处理统计MIoU精度

调用RefineNet_postprocess.py脚本推理结果与语义分割真值进行比对，可以获得IoU精度数据。
```bash
python3.7 ENet_postprocess.py --src-path=$datasets_path  --result-dir result/dumpOutput_device0_bs1/
```
第一个为真值所在目录，第二个为benchmark输出目录。  
查看输出结果：

```
miou: 54.620%
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上。

### 6.2 精度对比
ENet论文给出的精度是58.3%，但它没有训练代码，也没有给出训练好的模型权重。因此只能与910训练好的模型权重进行精度对比（0.54627）。

将得到的om离线模型推理miou精度与910训练好的`.pth权重的miou进行对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  

>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  

```
[e2e] throughputRate: 0.80452, latency: 621489
[data read] throughputRate: 199.855, moduleLatency: 5.00363
[preprocess] throughputRate: 138.372, moduleLatency: 7.22689
[infer] throughputRate: 3.89766, Interface throughputRate: 185.685, moduleLatency: 257.428
[post] throughputRate: 0.806308, moduleLatency: 1240.22
```
batch1 310单卡吞吐率： 742.740 
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  

```
[e2e] throughputRate: 0.805743, latency: 620546
[data read] throughputRate: 135.277, moduleLatency: 7.39224
[preprocess] throughputRate: 125.262, moduleLatency: 7.98329
[infer] throughputRate: 33.3087, Interface throughputRate: 170.983, moduleLatency: 30.4201
[post] throughputRate: 0.0516085, moduleLatency: 19376.7
```
batch16 310单卡吞吐率 ：683.932

>没有遇到性能不达标的问题，故不需要进行性能优化

