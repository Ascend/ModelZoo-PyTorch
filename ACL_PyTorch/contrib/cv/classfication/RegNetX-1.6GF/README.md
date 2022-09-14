# RegNetX-1.6GF Onnx模型端到端推理指导
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
	-   [5.1 ais_infer工具概述](#51-ais_infer工具概述)
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
[RegNetX-1.6GF论文](https://arxiv.org/abs/2003.13678)  

### 1.2 代码地址
[RegNetX-1.6GF代码](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/regnet.py)  
branch:master commit_id:742c2d524726d426ea2745055a5b217c020ccc72

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC1

torch == 1.10.2
torchvision == 0.11.3
onnx == 1.9.0
```

### 2.2 python第三方库

```
numpy == 1.22.3
Pillow == 9.0.1
opencv-python == 4.5.2.54
timm == 0.4.12
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.RegNetX-1.6GF模型代码在timm里，安装timm，arm下需源码安装，参考https://github.com/rwightman/pytorch-image-models
，若安装过程报错请百度解决
```
git clone https://github.com/rwightman/pytorch-image-models
cd pytorch-image-models
python3.7 setup.py install
cd ..
```

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

2.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 RegNetX_onnx.py regnetx_016-65ca972a.pth RegNetX-1.6GF.onnx
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
atc --framework=5 --model=./RegNetX-1.6GF.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=RegNetX-1.6GF_bs1 --log=debug --soc_version=Asend{chip_name}

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
python3.7 imagenet_torch_preprocess.py /home/dataset/ImageNet/ILSVRC2012_img_val ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 get_info.py bin ./prep_dataset ./RegNetX-1.6GF_prep_bin.info 224 224
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
## 5 离线推理

-   **[ais工具概述](#51-ais工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 ais_infer工具概述

ais_infer工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer
### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
```
python3.7.5 ais_infer.py --model /home/tangxiao/file/RegNetX-1.6GF_bs1.om --input "/home/tangxiao/RegNetX-1.6GF/prep_datase" --output "/home/tangxiao/RegNetX-1.6GF/result" --outfmt TXT
```
输出结果默认会建立日期+时间的子文件夹保存输出结果 如果指定output_dirname 将保存到output_dirname的子文件夹下，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

后处理统计TopN精度

调用vision_metric_ImageNet.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 vision_metric_ImageNet.py ./result/output_dirname/ opt/npu/val_label.txt ./ result_bs.json
```
第一个为ais_infer输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：
```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "76.93%"}, {"key": "Top2 accuracy", "value": "86.72%"}, {"key": "Top3 accuracy", "value": "90.25%"}, {"key": "Top4 accuracy", "value": "92.16%"}, {"key": "Top5 accuracy", "value": "93.42%"}]}
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上

### 6.2 开源TopN精度
[timm官网精度](https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv)
```
model	    top1	top1_err	top5	top5_err	param_count	img_size	cropt_pct	interpolation
regnetx_016	76.950	23.050	    93.420	6.580	    9.19	     224	    0.875	     bicubic
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
ais_infer工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用ais_infer纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准。模型的性能以使用ais_infer工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用ais_infer工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
1.benchmark工具在整个数据集上推理获得性能数据,在310上的性能
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  

batch1 310吞吐率：930.576fps
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  

batch16 310吞吐率：2783.012fps

batch4性能：./benchmark.x86_64 -round=20 -om_path=RegNetX-1.6GF_bs4.om -device_id=0 -batch_size=4
batch4 310吞吐率：3355.08fps  

batch8性能：
batch8 310吞吐率：3608.972fps  

batch32性能：
batch32 310吞吐率：3170.128fps  

batch64性能：
batch64 310吞吐率：3186.384fps

2.在310p上的性能

batch1性能：
batch1 310p吞吐率：1677.04fps 

batch4性能：
batch4 310p吞吐率：4372.53fps

batch8性能：
batch8 310p吞吐率：5486.04fps

batch16性能：
batch16 310p吞吐率：3934.81fps

batch32性能：
batch32 310p吞吐率：3752.48fps

batch64性能：
batch64 310p吞吐率：3623.42fps

### 7.2 T4性能数据
在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2  
batch1性能：
```
trtexec --onnx=RegNetX-1.6GF.onnx --fp16 --shapes=image:1x3x224x224 --threads

```
batch1 t4吞吐率：436.275fps  

batch16性能：
```
trtexec --onnx=RegNetX-1.6GF.onnx --fp16 --shapes=image:16x3x224x224 --threads

```
batch16 t4吞吐率：1867.646fps  

batch4性能：
batch4 t4吞吐率：1084.422fps  

batch8性能：
batch8 t4吞吐率：1532.781fps  

batch32性能：
batch32 t4吞吐率：2166.993fps  

batch64性能：
batch64 t4吞吐率：2276.486fps

### 7.3 性能对比
模型的所有batch_size都能满足在310p上的性能高于在310上的性能，同时在310p上的性能也能达到在t4性能的1.6倍以上
 
