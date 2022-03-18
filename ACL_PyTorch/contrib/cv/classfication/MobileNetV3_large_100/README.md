# MobileNetV3_large_100 Onnx模型端到端推理指导
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
[MobileNetV3论文](https://arxiv.org/pdf/1905.02244v5.pdf)


### 1.2 代码地址
[MobileNetV3代码](https://github.com/rwightman/gen-efficientnet-pytorch)  branch: master commit id:a4ac4dd7d72069a2aa4564df53ff4eb9b9f64400


## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  
-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.0.1

torch >= 1.5.0
torchvision >= 0.6.0
onnx >= 1.7.0
```

### 2.2 python第三方库

```
numpy == 1.18.5
Pillow == 7.2.0
opencv-python == 4.5.2.54
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  
-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.下载.pth权重文件  
[MobileNetV3预训练.pth权重文件](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth) 
文件md5sum：e5a1723b0a2ccdd058af3493100b4a93 

2.安装gen-efficientnet-pytorch

```
git clone https://github.com/rwightman/gen-efficientnet-pytorch.git
python3.7 -m pip install -e gen-efficientnet-pytorch
```

3.编写pth2onnx脚本MobileNetV3_pth2onnx.py

 **说明：**  
>目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件

```
python3.7 MobileNetV3_pth2onnx.py mobilenetv3_large_100_ra-f55367f5.pth mobilenetv3_100.onnx
```

 **模型转换要点：**  
>geffnet.create_model()中必须设置exportable=True，这样会把不能导出的HardSwishJitAutoFn算子用其他算子替代
>
>代码中的原始注解如下：
>
>> exportable=True flag disables autofn/jit scripted activations and uses Conv2dSameExport layers for models using SAME padding

### 3.2 onnx转om模型

1.设置环境变量

```
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)

```
atc --framework=5 --model=./mobilenetv3_100.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=mobilenetv3_100_bs16 --log=debug --soc_version=Ascend310
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  
-   **[数据集预处理](#42-数据集预处理)**  
-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
对于图像分类任务，该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/root/datasets/imagenet/val与/root/datasets/imagenet/val_label.txt

### 4.2 数据集预处理
1.预处理脚本img_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件

```
python3.7 img_preprocess.py /root/datasets/imagenet/val ./prep_dataset
```
第一个参数为验证集路径，第二个参数为预处理后生成的二进制文件的存储路径

### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```
python3.7 gen_dataset_info.py bin ./prep_dataset ./mobilenetv3_100_prep_bin.info 224 224
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
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=mobilenetv3_100_bs16.om -input_text_path=./mobilenetv3_100_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_devicex，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据。
```
python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ /root/datasets/imagenet/val_label.txt ./ result.json

```
第一个参数为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名，其中存有推理的Top5精度。
对batch1和batch16的模型分别调用benchmark进行推理，并统计其Top5的精度。查看其输出结果：

```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "75.77%"}, {"key": "Top2 accuracy", "value": "85.58%"}, {"key": "Top3 accuracy", "value": "89.29%"}, {"key": "Top4 accuracy", "value": "91.24%"}, {"key": "Top5 accuracy", "value": "92.53%"}]}
```
经过对bs1与bs16的om测试，本模型batch1与batch16的精度没有差别，精度数据均如上。

### 6.2 开源TopN精度
[github开源代码仓精度](https://rwightman.github.io/pytorch-image-models/results/)

```
Model                           Acc@1     Acc@5
mobilenetv3_large_100		75.766	  92.542
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  

>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时会统计性能数据，存储于result/perf_vision_batchsize_bs_device_0.txt中。但是推理整个数据集较慢，如此测性能时需要确保benchmark独占device，使用npu-smi info可以查看device是否空闲。
除此之外，也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具推理得到bs1与bs16的性能数据为准；对于使用benchmark工具测试的batch4，8，32的性能数据仅在README.md中作如下记录。  
1.benchmark工具在整个数据集上推理获得性能数据   
使用benchmark工具的纯推理功能测试模型的推理性能，命令如下：

```
./benchmark.x86_64 -round=20 -om_path=mobilenetv3_100_bs1.om -device_id=0 -batch_size=1
```
benchmark工具进行纯推理后测得的性能数据存储于result/PureInfer_perf_of_mobilenet-v1_bsx_in_device_0.txt，其中x为模型的batch_size。

batch1的性能，benchmark工具在整个数据集上推理后生成result/PureInfer_perf_of_mobilenetv3_100_bs1_in_device_0.txt：
```
[e2e] throughputRate: 230.705, latency: 216727
[data read] throughputRate: 244.812, moduleLatency: 4.08477
[preprocess] throughputRate: 244.525, moduleLatency: 4.08957
[infer] throughputRate: 232.04, Interface throughputRate: 378.955, moduleLatency: 3.62845
[post] throughputRate: 232.04, moduleLatency: 4.30961
```
Interface throughputRate: 378.955，378.955x4=1515.820即是batch1 310单卡吞吐率

batch16的性能，benchmark工具在整个数据集上推理后生成
result/PureInfer_perf_of_mobilenetv3_100_bs16_in_device_0.txt

```
[e2e] throughputRate: 139.214, latency: 359159
[data read] throughputRate: 140.692, moduleLatency: 7.10771
[preprocess] throughputRate: 140.553, moduleLatency: 7.11475
[infer] throughputRate: 140.577, Interface throughputRate: 945.665, moduleLatency: 2.51216
[post] throughputRate: 8.7859, moduleLatency: 113.819
```
Interface throughputRate: 945.665，945.665x4=3782.660即是batch16 310单卡吞吐率

batch4性能：
```
ave_throughputRate = 821.703samples/s, ave_latency = 1.245ms
```
batch4 310单卡吞吐率：821.703x4=3286.812 fps

batch8性能：
```
ave_throughputRate = 907.738samples/s, ave_latency = 1.11522ms
```
batch8 310单卡吞吐率：907.738x4=3630.952 fps

batch32性能：
```
ave_throughputRate = 823.648samples/s, ave_latency = 1.21733ms
```
batch32 310单卡吞吐率：823.648x4=3294.592 fps

 **性能优化：**  
>profiling工具分析，Conv2D、Add和TransData三个算子耗时最长
>
>未做性能优化
