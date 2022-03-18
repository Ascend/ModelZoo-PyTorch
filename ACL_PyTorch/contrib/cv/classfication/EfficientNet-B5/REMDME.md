
# EfficientNet-B5 模型端到端推理指导
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
[EfficientNet-B5论文](https://arxiv.org/abs/1905.11946)  

### 1.2 代码地址
[EfficientNet-B5代码](https://github.com/facebookresearch/pycls)  
branch:master  
commit_id:af463d014d259bf8483b981a57a2a85c10209252 

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

1.下载pth权重文件并改名为efficientnetb5.pyth  
[EfficientNet-B5预训练pth权重文件](https://dl.fbaipublicfiles.com/pycls/dds_baselines/161305138/EN-B5_dds_8gpu.pyth)  
文件md5sum: 8edbf8210066ba6646a99cadb5e47a41
```
wget https://dl.fbaipublicfiles.com/pycls/dds_baselines/161305138/EN-B5_dds_8gpu.pyth
mv EN-B5_dds_8gpu.pyth efficientnetb5.pyth
```
2.下载efficientnet源码，并将模型代码里/pycls/configs/dds_baselines/effnet/EN-B5_dds_8gpu.yaml复制到EfficientNet-B5文件夹并改名为efficientnetb5_dds_8gpu.yaml
```
git clone https://github.com/facebookresearch/pycls
cp ./pycls/configs/dds_baselines/effnet/EN-B5_dds_8gpu.yaml .
mv EN-B5_dds_8gpu.yaml efficientnetb5_dds_8gpu.yaml
```
3.编写efficientnetb5_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行efficientnetb5_pth2onnx脚本，生成onnx模型文件
```
python3.7 efficientnetb5_pth2onnx.py efficientnetb5.pyth efficientnetb5_dds_8gpu.yaml  efficientnetb5.onnx
```


### 3.2 onnx转om模型

1.设置环境变量
```
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1开发辅助工具指南 (推理) 01
```
atc --framework=5 --model=efficientnetb5.onnx --output=efficientnetb5_bs16 --input_format=NCHW --input_shape="image:16,3,456,456" --auto_tune_mode="RL,GA" --log=debug --soc_version=Ascend310
```

 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明
## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/opt/npu/imagent/val与/opt/npu/imagenet/val_label.txt。

### 4.2 数据集预处理
1.预处理脚本imagenet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3 imagenet_torch_preprocess.py efficientnetB5 /opt/npu/imagenet/val ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3 get_info.py bin ./prep_dataset ./efficientnetb5.info 456 456
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1推理benchmark工具用户指南 01
### 5.2 离线推理
1.设置环境变量
```
source env.sh
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=efficientnetb5_bs16.om -input_text_path=./efficientnetb5.info -input_width=456 -input_height=456 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_device{0}，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

后处理统计TopN精度

调用vision_metric_ImageNet.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3 vision_metric_ImageNet.py result/dumpOutput_device0/ /opt/npu/imagenet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：
```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "78.28%"}, {"key": "Top2 accuracy", "value": "87.67%"}, {"key": "Top3 accuracy", "value": "90.9%"}, {"key": "Top4 accuracy", "value": "92.65%"}, {"key": "Top5 accuracy", "value": "93.76%"}]}

```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上

### 6.2 开源TopN精度
[EfficientNet-B5官网精度](https://github.com/facebookresearch/pycls/blob/master/dev/model_error.json)
```
Model               Acc@1     Acc@5
EfficientNet-B5     78.326    93.762
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  

- 1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能（测试时使用了autotune），benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 28.9676, latency: 1.72607e+06
[data read] throughputRate: 29.4088, moduleLatency: 34.0035
[preprocess] throughputRate: 29.3509, moduleLatency: 34.0705
[infer] throughputRate: 28.9949, Interface throughputRate: 35.377, moduleLatency: 34.1232
[post] throughputRate: 28.9949, moduleLatency: 34.4888

```
Interface throughputRate: 35.377，35.377x4=141.508即是batch1 310单卡吞吐率。

- 2.batch16的性能（测试时使用了autotune），benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：
```
[e2e] throughputRate: 33.0042, latency: 1.51496e+06
[data read] throughputRate: 33.4934, moduleLatency: 29.8566
[preprocess] throughputRate: 33.4487, moduleLatency: 29.8966
[infer] throughputRate: 33.0372, Interface throughputRate: 41.2268, moduleLatency: 29.8799
[post] throughputRate: 2.06482, moduleLatency: 484.304

```
Interface throughputRate: 41.2268，41.2268x4=164.9072即是batch16 310单卡吞吐率。

- 3.batch4性能（测试时使用了autotune）：
```
./benchmark.x86_64 -round=20 -om_path=efficientnetB5_bs4.om -device_id=0 -batch_size=4
```
```
[INFO] Dataset number: 19 finished cost 99.801ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_efficientnetb5_bs4_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 39.9892samples/s, ave_latency: 25.092ms

```
batch4 310单卡吞吐率：39.9892x4=159.9568fps 
- 4.batch8性能（测试时使用了autotune）：
```
./benchmark.x86_64 -round=20 -om_path=efficientnetB5_bs8.om -device_id=0 -batch_size=8
```
```
[INFO] Dataset number: 19 finished cost 196.082ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_efficientnetb5_bs8_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 40.7769samples/s, ave_latency: 24.5726ms

```
batch8 310单卡吞吐率：40.7769x4=163.1076fps 
- 5.batch32性能（测试时使用了autotune）：
```
./benchmark.x86_64 -round=20 -om_path=efficientnetB5_bs32.om -device_id=0 -batch_size=32
```
```
[INFO] Dataset number: 19 finished cost 785.402ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_efficientnetb5_bs32_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 40.7859samples/s, ave_latency: 24.5273ms

```
batch32 310单卡吞吐率：40.7859x4=163.1436fps 

 **性能优化：**  
>对于bs16，不使用autotune的单卡吞吐率为148.5fps，使用autotune后单卡吞吐率为164.9fps。使用autotune后bs16性能达标
