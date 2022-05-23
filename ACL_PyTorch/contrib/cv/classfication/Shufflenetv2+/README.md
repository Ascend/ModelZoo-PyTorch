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
2.shufflenetv2+模型代码在代码仓里

```
github上Shufflenetv2+没有安装脚本，在pth2onnx脚本中引用代码仓定义的ShuffleNetv2+：

git clone https://github.com/megvii-model/ShuffleNet-Series.git
```
具体开源代码已放置在本目录下

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

3.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 shufflenetv2_pth2onnx.py ShuffleNetV2+.Small.pth.tar shufflenetv2_bs1.onnx 1
```
脚本第一参数为输入的pth模型，第二参数为输出的onnx模型名，第三参数为bs大小

shufflenetv2_pth2onnx_bs{x}.py是固定bs的转换脚本，为保留文件，不推荐使用

### 3.2 onnx转om模型

1.设置环境变量
```
source env.sh
```

 **说明：**  
>此处提供一份默认的env.sh脚本作为环境变量示例，具体设备需配置相应环境变量

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01
```
atc --framework=5 --model=./shufflenetv2_bs1.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=shufflenetv2_bs1 --log=debug --soc_version=Ascend310
```
针对不同bs的onnx模型，需修改--model, --input_shape, --output 三个参数中的bs值
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
source env.sh
```
2.执行离线推理

```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=shufflenetv2_bs1.om -input_text_path=./shufflenetv2_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_devicex，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

后处理统计TopN精度

调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ /path/to/imagenet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  


### 6.2 开源TopN精度
[开源代码仓精度](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2%2B)

```
Model                     Acc@1     Acc@5
shufflenetv2_open_source  74.1      91.7
shufflenetv2_bs32_310	  74.08	    91.67
shufflenetv2_bs32_310P	  74.08	    91.67
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
