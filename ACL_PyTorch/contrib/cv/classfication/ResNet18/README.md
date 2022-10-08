# ResNet18 Onnx模型端到端推理指导
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
[ResNet18论文](https://arxiv.org/pdf/1512.03385.pdf)  

### 1.2 代码地址
[ResNet18代码](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)  
branch:master
commit_id:7d955df73fe0e9b47f7d6c77c699324b256fc41f

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.0.1

torch == 1.5.1
torchvision == 0.6.1
onnx == 1.9.0
```

### 2.2 python第三方库

```
numpy == 1.19.2
Pillow == 8.2.0
opencv-python == 4.5.2
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
[ResNet-18预训练pth权重文件](https://download.pytorch.org/models/resnet18-f37072fd.pth)  
```
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
```
文件MD5sum：e0b1c919e74f9a193d36871d9964bf7d

2.ResNet18模型代码在torchvision里，安装torchvision，arm下需源码安装，参考torchvision官网，若安装过程报错请百度解决
```
git clone https://github.com/pytorch/vision
cd vision
python3.7 setup.py install
cd ..
```
3.编写pth2onnx脚本resnet18_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 resnet18_pth2onnx.py ./resnet18-f37072fd.pth resnet18.onnx
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
atc --framework=5 --model=./resnet18.onnx --output=resnet18_bs1 --input_format=NCHW --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend310

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
python3.7 gen_dataset_info.py bin ./prep_dataset ./resnet18_prep_bin.info 224 224
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
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=resnet18_bs1.om -input_text_path=./resnet18_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
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
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "69.74%"}, {"key": "Top2 accuracy", "value": "80.49%"}, {"key": "Top3 accuracy", "value": "84.96%"}, {"key": "Top4 accuracy", "value": "87.38%"}, {"key": "Top5 accuracy", "value": "89.09%"}]}
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上

### 6.2 开源TopN精度
[torchvision官网精度](https://pytorch.org/vision/stable/models.html)
```
Model        Acc@1     Acc@5
ResNet-18    69.758    89.078
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 336.39, latency: 148637
[data read] throughputRate: 357.787, moduleLatency: 2.79496
[preprocess] throughputRate: 357.147, moduleLatency: 2.79997
[infer] throughputRate: 338.442, Interface throughputRate: 787.709, moduleLatency: 2.21785
[post] throughputRate: 338.44, moduleLatency: 2.95473
```
Interface throughputRate: 787.709，787.709x4=3150.836既是batch1 310单卡吞吐率  
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  
```
[e2e] throughputRate: 355.715, latency: 140562
[data read] throughputRate: 377.215, moduleLatency: 2.65101
[preprocess] throughputRate: 376.578, moduleLatency: 2.65549
[infer] throughputRate: 357.628, Interface throughputRate: 1509.64, moduleLatency: 2.1165
[post] throughputRate: 22.3509, moduleLatency: 44.7409
```
Interface throughputRate: 1509.64，1509.64x4=6038.56既是batch16 310单卡吞吐率  
batch4性能：
```
[e2e] throughputRate: 218.705, latency: 228619
[data read] throughputRate: 219.993, moduleLatency: 4.5456
[preprocess] throughputRate: 219.699, moduleLatency: 4.55169
[infer] throughputRate: 219.574, Interface throughputRate: 1103.24, moduleLatency: 2.38868
[post] throughputRate: 54.8929, moduleLatency: 18.2173
```
batch4 310单卡吞吐率：1103.24x4=4412.96fps  
batch8性能：
```
[e2e] throughputRate: 175.032, latency: 285662
[data read] throughputRate: 175.909, moduleLatency: 5.68474
[preprocess] throughputRate: 175.703, moduleLatency: 5.69143
[infer] throughputRate: 175.795, Interface throughputRate: 1446.02, moduleLatency: 2.17869
[post] throughputRate: 21.9741, moduleLatency: 45.5081
```
batch8 310单卡吞吐率：1446.02x4=5784.08fps  
batch32性能：
```
[e2e] throughputRate: 151.68, latency: 329642
[data read] throughputRate: 152.292, moduleLatency: 6.56634
[preprocess] throughputRate: 152.082, moduleLatency: 6.57541
[infer] throughputRate: 152.081, Interface throughputRate: 1375.46, moduleLatency: 2.20383
[post] throughputRate: 4.75395, moduleLatency: 210.352
```
batch32 310单卡吞吐率：1375.46x4=5501.84fps  

 **性能优化：**  
>没有遇到性能不达标的问题，故不需要进行性能优化

