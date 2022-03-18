# Wide_ResNet101_2 Onnx模型端到端推理指导

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
	-   [6.1 离线推理精度统计](#61-离线推理精度统计)
	-   [6.2 开源精度](#62-开源精度)
	-   [6.3 精度对比](#63-精度对比)
	
- [7 性能对比](#7-性能对比)

  -   [7.1 npu性能数据](#71-npu性能数据)

  



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址

[Wide_ResNet论文](https://arxiv.org/pdf/1605.07146.pdf)  

### 1.2 代码地址

[Wide_ResNet代码](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)

branch:master
commit id:7d955df73fe0e9b47f7d6c77c699324b256fc41f

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
opencv-python == 4.2.0.34
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1. 下载pth权重文件  

[wrn101_2权重文件下载](https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth)

文件md5sum:  5961435974bb43104b5a3180fea7c2c4 

```
wget https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth
```



2. 下载模型代码

```
git clone https://github.com/pytorch/vision
cd vision
git reset 7d955df73fe0e9b47f7d6c77c699324b256fc41f --hard
python3.7 setup.py install
cd ..
```

3. 编写pth2onnx脚本wrn101_2_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4. 执行pth2onnx脚本，生成onnx模型文件


```python
python3.7 wrn101_2_pth2onnx.py wide_resnet101_2-32ee1156.pth wrn101_2_pth.onnx
```

 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明

### 3.2 onnx转om模型

1.设置环境变量

```python
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01

```python
atc --framework=5 --model=wrn101_2_pth.onnx --output=wrn101_2_bs16 --input_format=NCHW --input_shape="image:16,3,224,224" --log=debug --soc_version=Ascend310
```



## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/opt/npu/imagenet/val与/opt/npu/imagenet/val_label.txt

### 4.2 数据集预处理

1.预处理脚本imagenet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件

```
python3.7 imagenet_torch_preprocess.py resnet /opt/npu/imagenet/val ./prep_dataset
```
### 4.3 生成数据集信息文件

1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```python
python3.7 gen_dataset_info.py bin ./prep_dataset ./wrn101_2_prep_bin.info 224 224
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息

## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN V100R020C10 推理benchmark工具用户指南 01

### 5.2 离线推理

1.设置环境变量

```
source env.sh
```
2.执行离线推理

```python
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=wrn101_2_bs16.om -input_text_path=./wrn101_2_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_device{0}，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

后处理统计TopN精度

调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ /opt/npu/imagenet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：

```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "78.84%"}, {"key": "Top2 accuracy", "value": "88.41%"}, {"key": "Top3 accuracy", "value": "91.66%"}, {"key": "Top4 accuracy", "value": "93.26%"}, {"key": "Top5 accuracy", "value": "94.29%"}]}
```

### 6.2 开源精度

[torchvision官网精度](https://pytorch.org/vision/stable/models.html)

```
Model                 Acc@1       Acc@5
wide_resnet101_2      78.848     94.284
```
### 6.3 精度对比

将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，统计结果如下。精度下降在1%范围之内，故精度达标。  

```
                Acc@1      Acc@5
bs1             78.84      94.29
bs16            78.85      94.29
```

 **精度调试：**  

>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据

benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。

1.benchmark工具在整个数据集上推理获得性能数据

batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：

```
[e2e] throughputRate: 105.142, latency: 475550
[data read] throughputRate: 111.355, moduleLatency: 8.98031
[preprocess] throughputRate: 111.053, moduleLatency: 9.00469
[infer] throughputRate: 105.494, Interface throughputRate: 127.878, moduleLatency: 8.77965
[post] throughputRate: 105.494, moduleLatency: 9.47924
```

Interface throughputRate: 127.878，127.878x4=511.512即是batch1 310单卡吞吐率

batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：

```
[e2e] throughputRate: 117.321, latency: 426182
[data read] throughputRate: 124.66, moduleLatency: 8.0218
[preprocess] throughputRate: 124.054, moduleLatency: 8.06101
[infer] throughputRate: 117.825, Interface throughputRate: 169.604, moduleLatency: 7.35524
[post] throughputRate: 7.36397, moduleLatency: 135.796
```

Interface throughputRate: 169.604，169.604x4=678.416即是batch1 310单卡吞吐率

batch4性能：

```
[e2e] throughputRate: 114.374, latency: 437161
[data read] throughputRate: 121.259, moduleLatency: 8.2468
[preprocess] throughputRate: 121.014, moduleLatency: 8.26352
[infer] throughputRate: 114.92, Interface throughputRate: 157.07, moduleLatency: 7.83108
[post] throughputRate: 28.73, moduleLatency: 34.8068
```

batch4 310单卡吞吐率：157.07x4=628.28fps
batch8性能：

```
[e2e] throughputRate: 111.341, latency: 449071
[data read] throughputRate: 117.759, moduleLatency: 8.49194
[preprocess] throughputRate: 117.55, moduleLatency: 8.50701
[infer] throughputRate: 111.703, Interface throughputRate: 156.132, moduleLatency: 7.85466
[post] throughputRate: 13.9628, moduleLatency: 71.6188
```

batch8 310单卡吞吐率：156.132x4=624.528fps
batch32性能：

```
[e2e] throughputRate: 102.387, latency: 488344
[data read] throughputRate: 108.61, moduleLatency: 9.20728
[preprocess] throughputRate: 108.389, moduleLatency: 9.22602
[infer] throughputRate: 102.81, Interface throughputRate: 139.595, moduleLatency: 8.59119
[post] throughputRate: 3.2138, moduleLatency: 311.159
```

batch32 310单卡吞吐率：139.595x4=558.38fps

**性能优化：**

> 对于batch32的性能不达标，从profiling数据的op_statistic_0_1.csv看出影响性能的是Conv2D算子，从op_summary_0_1.csv看出单个Conv_Relu算子耗时0.6毫秒到12毫秒，shape大的耗时就多，不存在优化问题
