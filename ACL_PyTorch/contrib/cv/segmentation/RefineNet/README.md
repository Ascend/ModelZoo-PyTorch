# RefineNet Onnx模型端到端推理指导
- [RefineNet Onnx模型端到端推理指导](#refinenet-onnx模型端到端推理指导)
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
[RefineNet论文](https://arxiv.org/pdf/1611.06612.pdf)  

### 1.2 代码地址
[light-weight-refinenet代码](https://github.com/DrSleep/light-weight-refinenet)  
branch:master  
commit_id: 538fe8b39327d8343763b859daf7b9d03a05396e  
备注：commit_id是指基于该次提交时的模型代码做推理，通常选择稳定版本的最后一次提交，或代码仓最新的一次提交  

[RefineNet模型代码](https://github.com/DrSleep/refinenet-pytorch)

branch:master  
commit_id: 8f25c076016e61a835551493aae303e81cf36c53

[RefineNet(in Pytorch)](https://github.com/DrSleep/refinenet-pytorch)这个仓库的代码只给出了模型代码，没有给出训练代码，因此RefineNet的训练流程使用了该作者的另一个仓库https://github.com/DrSleep/light-weight-refinenet的训练代码搭配RefineNet的模型代码。

上述两个开源代码仓库都没有给出训练好的模型权重文件，因此使用910训练好的pth权重文件来做端到端推理，该权重文件的精度是**0.7844**。

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC1
pytorch >= 1.5.0
torchvision >= 0.6.0
onnx >= 1.7.0
```

### 2.2 python第三方库

```
numpy == 1.21.6
Pillow == 9.1.0
opencv-python == 4.1.0.25
albumentations == 1.1.0
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

1.RefineNet模型代码下载

```bash
git clone https://github.com/DrSleep/refinenet-pytorch.git RefineNet_pytorch
```
2.在npu上训练模型，稍微修改了下模型代码，需要把代码移植到开源模型代码中：

```
cd RefineNet_pytorch
git apply ../RefineNet.patch
cd ..
```

3.编写pth2onnx脚本RefineNet_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11


4.通过将densetorch包下载到本地安装

```bash
git clone https://github.com/drsleep/densetorch.git
cd densetorch
pip install -e .
```

5.执行pth2onnx脚本，生成onnx模型文件

```bash
python3.7 RefineNet_pth2onnx.py --input-file model/RefineNet_910.pth.tar --output-file model/RefineNet_910.onnx
```

### 3.2 onnx转om模型

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.1.RC1 开发辅助工具指南 (推理) 01

${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```BASH
atc --framework=5 --model=model/RefineNet_910.onnx --output=model/RefineNet_910_bs1 --input_format=NCHW --input_shape="input:1,3,500,500" --log=debug --soc_version=Ascend${chip_name}
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用SBD的5623张训练图片以及VOC2012的1464张训练图片作为训练集，VOC2012的1449张验证图片作为验证集。推理部分只需要用到这1449张验证图片，图片id存放在`VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt`，验证集输入图片存放在`VOCdevkit/VOC2012/JPEGImages`，验证集label存放在`VOCdevkit/VOC2012/SegmentationClass`。

下载VOC2012数据集后，把VOCdevkit文件夹放在`/opt/npu`下。

### 4.2 数据集预处理
1.参考开源代码仓库对验证集所做的预处理，编写预处理脚本。由于om模型需要固定输入的hw维，而原来的预处理方式不会更改验证集的输入图片的大小，因此需要把验证集图片按长边缩放到一个固定值，然后再把短边padding到一个固定值。根据打印出来的验证集图片的shape，h和w的固定值都选为500。使用这种预处理方式对原`.pth.tar`模型权重进行精度测试，得到的miou是0.7861，反而更高一些。

2.执行预处理脚本，生成数据集预处理后的bin文件

```bash
mkdir prepare_dataset
python3.7 RefineNet_preprocess.py --root-dir /opt/npu/VOCdevkit/VOC2012 --bin-dir ./prepare_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```bash
python3.7 get_info.py bin prepare_dataset ./refinenet_prep_bin.info 500 500
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息  
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.1.RC1 推理benchmark工具用户指南 01
### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
```bash
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=model/RefineNet_910_bs1.om \
    -input_text_path=./refinenet_prep_bin.info -input_width=500 -input_height=500 -output_binary=False -useDvpp=False
```

输出结果默认保存在当前目录result/dumpOutput_deviceX(X为对应的device_id)，每个输入对应一个_X.bin文件的输出。

## 6 精度对比

-   **[离线推理IoU精度](#61-离线推理IoU精度)**    
-   **[精度对比](#62-精度对比)**  

### 6.1 离线推理IoU精度

后处理统计IoU精度

调用RefineNet_postprocess.py脚本推理结果与语义分割真值进行比对，可以获得IoU精度数据。
```bash
ulimit -n 10240
python3.7 RefineNet_postprocess.py --val-dir /opt/npu --result-dir result/dumpOutput_device0
```
第一个为真值所在目录，第二个为benchmark输出目录。  
查看输出结果：

```
miou: 0.786359
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上。

### 6.2 精度对比
light-weight-refinenet开源代码仓库给出的精度是~76%，但使用的是light-weight-refinenet；而refinenet-pytorch开源代码仓库给出的精度是80.5%，但它没有训练代码，也没有给出训练好的模型权重。因此只能与910训练好的模型权重进行精度对比（0.7861）。

将得到的om离线模型推理miou精度与910训练好的`.pth.tar`权重的miou进行对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  

>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32，64的性能数据在README.md中如下作记录即可。  

1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  

```
[inference] throughputRate: 60.5672, Interface throughputRate: 91.434, moduleLatency: 16.1848
```
Interface throughputRate: 91.434是batch1 310P单卡吞吐率  
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  

```
[inference] throughputRate: 56.6365, Interface throughputRate: 77.7378, moduleLatency: 17.4284
```
Interface throughputRate: 77.7378是batch16 310P单卡吞吐率 


2.npu纯推理性能

batch1的性能，执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果

```bash
./benchmark.x86_64 -round=20 -om_path=model/RefineNet_910_bs1.om -device_id=0 -batch_size=1

[INFO] PureInfer result saved in ./result/PureInfer_perf_of_RefineNet_910_bs1_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 92.2434samples/s, ave_latency: 11.171ms
----------------------------------------------------------------
```

batch4的性能，执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果

```bash
./benchmark.x86_64 -round=20 -om_path=model/RefineNet_910_bs4.om -device_id=0 -batch_size=4

[INFO] PureInfer result saved in ./result/PureInfer_perf_of_RefineNet_910_bs4_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 88.0568samples/s, ave_latency: 11.4838ms
----------------------------------------------------------------
```

batch8的性能，执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果

```bash
./benchmark.x86_64 -round=20 -om_path=model/RefineNet_910_bs8.om -device_id=0 -batch_size=8

[INFO] PureInfer result saved in ./result/PureInfer_perf_of_RefineNet_910_bs8_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 84.4253samples/s, ave_latency: 11.8672ms
----------------------------------------------------------------
```

batch16的性能，执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果

```bash
./benchmark.x86_64 -round=20 -om_path=model/RefineNet_910_bs16.om -device_id=0 -batch_size=16

[INFO] PureInfer result saved in ./result/PureInfer_perf_of_RefineNet_910_bs16_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 80.3815samples/s, ave_latency: 12.4635ms
----------------------------------------------------------------
```

batch32的性能，执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果

```bash
./benchmark.x86_64 -round=20 -om_path=model/RefineNet_910_bs32.om -device_id=0 -batch_size=32

[INFO] PureInfer result saved in ./result/PureInfer_perf_of_RefineNet_910_bs32_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 79.9769samples/s, ave_latency: 12.5081ms
----------------------------------------------------------------
```

batch64的性能，执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果

```bash
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_RefineNet_910_bs64_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 79.1814samples/s, ave_latency: 12.6375ms
----------------------------------------------------------------
```

 **性能优化：**  

>没有遇到性能不达标的问题，故不需要进行性能优化

