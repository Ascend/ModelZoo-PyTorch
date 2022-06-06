Shufflenetv2+ Onnx模型端到端推理指导

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



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[shufflenetv2论文](https://arxiv.org/abs/1807.11164)  

### 1.2 代码地址
[shufflenetv2+代码](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2%2B)  
branch:master  
commit_id:d69403d4b5fb3043c7c0da3c2a15df8c5e520d89

具体开源代码已放置在本目录下

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN == 5.1.RC1
onnx == 1.11.0
pytorch == 1.11.0
torchvision == 0.6.0
```

### 2.2 python第三方库

```
numpy == 1.21.6
pillow == 8.3.2
opencv-python == 4.5.3.56
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
[shufflenetv2+预训练pth权重文件](https://pan.baidu.com/share/init?surl=EUQVoFPb74yZm0JWHKjFOw)  
文件md5sum: 1d6611049e6ef03f1d6afa11f6f9023e  

```
https://pan.baidu.com/share/init?surl=EUQVoFPb74yZm0JWHKjFOw  提取码：mc24
```
**注意：**
> 预训练pth权重文件的文件名为ShuffleNetV2+.Small.pth.tar，无需修改，在后续流程中直接使用

2.shufflenetv2+模型代码在代码仓里

```
github上Shufflenetv2+没有安装脚本，在pth2onnx脚本中引用代码仓定义的ShuffleNetv2+：

git clone https://github.com/megvii-model/ShuffleNet-Series.git
```

 **说明：**  
>具体开源代码已放置在本目录下

>注意目前ATC支持的onnx算子版本为11

3.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 shufflenetv2_pth2onnx.py ShuffleNetV2+.Small.pth.tar shufflenetv2_bs1.onnx 1
```
脚本第一参数为输入的pth模型，第二参数为输出的onnx模型名，第三参数为bs大小；

修改bs大小时，也需要修改对应的输出onnx模型的命名；目前已通过测试的bs为1，4，8，16，32，64;

**注意：**
>动态batch的onnx转换om失败并且测的性能数据也不对，每个batch的om需要对应batch的onnx来转换，每个batch的性能数据需要对应batch的模型来测试

### 3.2 onnx转om模型

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

 **说明：**  
>环境变量影响atc命令是否成功，在测试时如报错需验证环境变量的正确性

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01
```
atc --framework=5 --model=./shufflenetv2_bs1.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=shufflenetv2_bs1 --log=debug --soc_version=Ascend310
```
针对不同bs的onnx模型，需修改--model, --input_shape, --output 三个参数中的bs值;

针对不同的芯片(310/310P)，需修改参数--soc_version，分别为Ascend310、Ascend710;
## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签存放路径分别为/path/to/imagenet/val与/path/to/imagenet/val_label.txt

### 4.2 数据集预处理
1.预处理脚本imagenet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 imagenet_torch_preprocess.py /path/to/imagenet/val ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```
python3.7 get_info.py bin ./prep_dataset ./shufflenetv2_prep_bin.info 224 224
```
第一个参数为生成的bin文件路径，第二个为输出的info文件，后面为宽高信息
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310/310P上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理benchmark工具用户指南 01
### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理

```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=shufflenetv2_bs1.om -input_text_path=./shufflenetv2_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_device{x}，模型只有一个名为class的输出，shape为bs * 1000，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件;

对不同bs的模型进行测试时，需要将参数--batch_size修改为对应的bs，并将--om_path修改为对应bs的om模型路径；


## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

后处理统计TopN精度

调用vision_metric_ImageNet.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ /path/to/imagenet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  


### 6.2 开源TopN精度
[开源代码仓精度](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2%2B)

测试得到精度：
```
Model                     Acc@1     Acc@5
shufflenetv2_open_source  74.1      91.7
shufflenetv2_bs32_310	  74.08	    91.67
shufflenetv2_bs32_310P	  74.08	    91.67
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device。为快速获取性能数据，也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准。这里给出两种方式，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 127.82, latency: 391174
[data read] throughputRate: 137.318, moduleLatency:7.28236 
[preprocess] throughputRate: 136.812, moduleLatency:7.30929 
[infer] throughputRate: 129.743, Interface throughputRate: 165.945, moduleLatency: 6.96608
[post] throughputRate: 129.743, moduleLatency: 7.70755 
```
Interface throughputRate: 165.945，165.945x4=663.78既是batch1 310单卡吞吐率  
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  

```
[e2e] throughputRate: 135.666, latency: 368553
[data read] throughputRate: 136.006, moduleLatency: 7.35259
[preprocess] throughputRate: 135.87, moduleLatency: 7.35998
[infer] throughputRate: 135.947, Interface throughputRate: 711.064, moduleLatency: 2.83882
[post] throughputRate: 8.49641, moduleLatency: 117.697
```
Interface throughputRate: 711.064，711.064x4=2844.256既是batch16 310单卡吞吐率  
batch4性能：

```
[e2e] throughputRate: 145.107, latency: 344574
[data read] throughputRate: 146.001, moduleLatency: 6.84925
[preprocess] throughputRate: 145.801, moduleLatency: 6.85866
[infer] throughputRate: 145.752, Interface throughputRate: 479.134, moduleLatency: 3.5695
[post] throughputRate: 36.4377, moduleLatency: 27.4441
```
batch4 310单卡吞吐率：479.134x4=1916.546fps  
batch8性能：

```
[e2e] throughputRate: 137.536, latency: 363541
[data read] throughputRate: 138.134, moduleLatency: 7.23933
[preprocess] throughputRate: 137.957, moduleLatency: 7.24864
[infer] throughputRate: 137.935, Interface throughputRate: 625.442, moduleLatency: 3.06848
[post] throughputRate: 17.2415, moduleLatency: 57.9994
```
batch8 310单卡吞吐率：625.442x4=2501.768fps  
batch32性能：

```
[e2e] throughputRate: 122.976, latency: 406582
[data read] throughputRate: 123.332, moduleLatency: 8.10819
[preprocess] throughputRate: 123.204, moduleLatency: 8.11661
[infer] throughputRate: 123.232, Interface throughputRate: 667.022, moduleLatency: 2.93736
[post] throughputRate: 3.8521, moduleLatency: 259.599
```
batch32 310单卡吞吐率：667.022x4=2668.088fps  

**性能优化：**  

>没有遇到性能不达标的问题，故不需要进行性能优化