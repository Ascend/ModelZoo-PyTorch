vovnet39 Onnx模型端到端推理指导

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
[vovnet39论文](https://arxiv.org/abs/1904.09730)  

### 1.2 代码地址
[vovnet39代码](https://github.com/AlexanderBurkhart/cnn_train/tree/505637bcd08021e144c94e81401af6bc71fd46c6/VoVNet.pytorch/models_vovnet)  
branch:master  
commit_id:505637bcd08021e144c94e81401af6bc71fd46c6

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC1
pytorch == 1.5.0
torchvision == 0.6.0
onnx == 1.7.0
```

### 2.2 python第三方库

```
numpy == 1.18.5
Pillow == 7.2.0
opencv-python == 4.5.1.48
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
[vovnet39预训练pth权重文件](https://www.dropbox.com/s/1lnzsgnixd8gjra/vovnet39_torchvision.pth)  
文件md5sum: 23717a6cadd9729a704f894381444237 

```
http://www.dropbox.com/s/1lnzsgnixd8gjra/vovnet39_torchvision.pth
```
2.vovnet39模型代码在代码仓里

```
github上vovnet39没有安装脚本，在pth2onnx脚本中引用代码仓定义的vovnet39：

git clone https://github.com/AlexanderBurkhart/cnn_train.git
```

3.编写pth2onnx脚本vovnet39_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 vovnet39_pth2onnx.py vovnet39_torchvision.pth vovnet39.onnx
```
 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明  
### 3.2 onnx转om模型

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.1.RC1  开发辅助工具指南 (推理) 01
```
atc --framework=5 --model=./vovnet39.onnx --input_format=NCHW --input_shape="image:32,3,224,224" --output=vovnet39_bs32 --log=debug --soc_version=Ascend${chip_name}
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/opt/npu/imagenet/val与/root/datasets/imagenet/val_label.txt。

### 4.2 数据集预处理
1.预处理脚本vovnet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 vovnet_torch_preprocess.py /opt/npu/imagenet/val ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```
python3.7 get_info.py bin ./prep_dataset ./vovnet_prep_bin.info 224 224
```
第一个参数为生成的数据集文件格式，第二个为预处理后的数据文件路径，第三个参数为生成的数据集文件保存的路径，第四个和第五个参数分别为模型输入的宽度和高度
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.1.RC1  推理benchmark工具用户指南 01
### 5.2 离线推理
1.设置环境变量
```
source /usr/loca/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理

```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=32 -om_path=vovnet39_bs32.om -input_text_path=./vovnet_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_devicex，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

后处理统计TopN精度

调用vision_metric_ImageNet.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ /opt/npu/imagenet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看310P输出结果：
```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value
": "76.78%"}, {"key": "Top2 accuracy", "value": "86.6%"}, {"key": "Top3 accuracy", "value": "90.23%"}, {"key": "Top4 accuracy", "value": "92.22%"}, {"key": "Top5 accuracy", "value": "93.43%"}]}
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上

### 6.2 开源TopN精度
[开源代码仓精度](https://github.com/AlexanderBurkhart/cnn_train/tree/505637bcd08021e144c94e81401af6bc71fd46c6/VoVNet.pytorch)

```
Model               Acc@1     Acc@5
vovnet39	        76.77     93.43
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device。为快速获取性能数据，也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准。这里给出两种方式，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 344.666, latency: 145068
[data read] throughputRate: 386.506, moduleLatency: 2.58728
[preprocess] throughputRate: 384.54, moduleLatency: 2.60051
[inference] throughputRate: 347.876, Interface throughputRate: 838.101, moduleLatency: 2.3685
[postprocess] throughputRate: 347.877, moduleLatency: 2.87458
```
Interface throughputRate: 838.101，838.101既是batch1 310P单卡吞吐率  
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  

```
[e2e] throughputRate: 97.7301, latency: 511613
[data read] throughputRate: 98.2159, moduleLatency: 10.1816
[preprocess] throughputRate: 97.9543, moduleLatency: 10.2088
[inference] throughputRate: 98.0668, Interface throughputRate: 1335.53, moduleLatency: 6.6059
[postprocess] throughputRate: 6.131, moduleLatency: 163.106
```
Interface throughputRate: 1355.53，1355.53既是batch16 310P单卡吞吐率  
batch4性能：

```
[e2e] throughputRate: 147.359, latency: 339307
[data read] throughputRate: 149.622, moduleLatency: 6.68349
[preprocess] throughputRate: 149.093, moduleLatency: 6.70722
[inference] throughputRate: 148.151, Interface throughputRate: 1702.79, moduleLatency: 4.64194
[postprocess] throughputRate: 37.0399, moduleLatency: 26.9979
```
batch4 310P单卡吞吐率：1702.79fps  
batch8性能：

```
[e2e] throughputRate: 104.855, latency: 476848
[data read] throughputRate: 105.421, moduleLatency: 9.48575
[preprocess] throughputRate: 105.073, moduleLatency: 9.51719
[inference] throughputRate: 105.229, Interface throughputRate: 1334.48, moduleLatency: 6.59057
[postprocess] throughputRate: 13.1556, moduleLatency: 76.0132
```
batch8 310P单卡吞吐率：1334.48fps  
batch32性能：

```
[e2e] throughputRate: 84.5401, latency: 591435
[data read] throughputRate: 84.7536, moduleLatency: 11.7989
[preprocess] throughputRate: 84.6081, moduleLatency: 11.8192
[inference] throughputRate: 84.7406, Interface throughputRate: 1408.41, moduleLatency: 7.33793
[postprocess] throughputRate: 2.65064, moduleLatency: 377.268
```
batch32 310P单卡吞吐率：1408.41fps  

batch64性能：

```
[e2e] throughputRate: 160.435, latency: 311653
[data read] throughputRate: 161.63, moduleLatency: 6.18695
[preprocess] throughputRate: 161.124, moduleLatency: 6.2064
[inference] throughputRate: 161.372, Interface throughputRate: 1557.35, moduleLatency: 2.79961
[postprocess] throughputRate: 2.52698, moduleLatency: 395.729
```
batch64 310P单卡吞吐率：1557.35fps 


 **性能优化：**  

>没有遇到性能不达标的问题，故不需要进行性能优化
