# SE_ResNet50 Onnx模型端到端推理指导
- [SE_ResNet50 Onnx模型端到端推理指导](#se_resnet50-onnx模型端到端推理指导)
	- [1 模型概述](#1-模型概述)
		- [1.1 论文地址](#11-论文地址)
		- [1.2 代码地址](#12-代码地址)
	- [2 环境说明](#2-环境说明)
		- [2.1 推理硬件设备](#21-推理硬件设备)
		- [2.2 深度学习框架](#22-深度学习框架)
		- [2.3 Python第三方库](#23-python第三方库)
	- [3 模型转换](#3-模型转换)
		- [3.1 获取pth权重文件](#31-获取pth权重文件)
		- [3.2 pth转onnx模型](#32-pth转onnx模型)
		- [3.3 onnx转om模型](#33-onnx转om模型)
	- [4 数据集预处理](#4-数据集预处理)
		- [4.1 数据集获取](#41-数据集获取)
		- [4.2 数据集预处理](#42-数据集预处理)
		- [4.3 生成数据集信息文件](#43-生成数据集信息文件)
	- [5 离线推理](#5-离线推理)
		- [5.1 benchmark工具概述](#51-benchmark工具概述)
		- [5.2 离线推理](#52-离线推理)
		- [5.3 性能验证](#53-性能验证)
	- [6 评测结果](#6-评测结果)
	- [6 test目录说明](#6-test目录说明)

## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[SE_ResNet50论文](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)  

### 1.2 代码地址
[SE_ResNet50代码](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py)  

## 2 环境说明

-   **[推理硬件设备](#21-推理硬件设备)**  

-   **[深度学习框架](#22-深度学习框架)**  

-   **[Python第三方库](#23-Python第三方库)**  

### 2.1 推理硬件设备
```
Ascend310P
```

### 2.2 深度学习框架
```
CANN 5.0.4

torch == 1.8.0
torchvision == 0.9.0
onnx == 1.10.2
```

### 2.3 Python第三方库

```
numpy == 1.21.4
opencv-python == 4.5.4.58
pretrainedmodels == 0.7.4
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[获取pth权重文件](#31-获取pth权重文件)**  

-   **[pth转onnx模型](#32-pth转onnx模型)**  

-   **[onnx转om模型](#33-onnx转om模型)**  

### 3.1 获取pth权重文件
执行命令：

```
wget https://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth
```
执行后在当前目录下获取pth权重文件：se_resnet50-ce0d4300.pth。

### 3.2 pth转onnx模型
执行命令：

```
python3 SE_ResNet50_pth2onnx.py ./se_resnet50-ce0d4300.pth ./se_resnet50_dynamic_bs.onnx
```

命令参数分别为输入pth文件：./se_resnet50-ce0d4300.pth和输出onnx文件：./se_resnet50_dynamic_bs.onnx  
执行后在当前路径下生成se_resnet50_dynamic_bs.onnx模型文件。  

### 3.3 onnx转om模型

a.设置环境变量：

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

该命令中使用CANN默认安装路径(/usr/local/Ascend/ascend-toolkit)中的环境变量，使用过程中请按照实际安装路径设置环境变量。

b.执行atc模型转换命令：

${chip_name}可通过`npu-smi info`指令查看，例：310P3

![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```
atc --model=./se_resnet50_dynamic_bs.onnx --framework=5 --input_format=NCHW --input_shape="image:32,3,224,224" --output=./se_resnet50_fp16_bs32 --log=error --soc_version=Ascend${chip_name} --insert_op_conf=./aipp_SE_ResNet50_pth.config --enable_small_channel=1
```

参数说明：
    --model：为ONNX模型文件。  
    --framework：5代表ONNX模型。  
    --input_format：输入数据的格式。  
    --input_shape：输入数据的shape。  
    --output：输出的OM模型。  
    --log：日志级别。  
    --soc_version：处理器型号。  
    --insert_op_config：插入算子的配置文件路径与文件名，例如aipp预处理算子。  
    --enable_small_channel：Set enable small channel. 0(default): disable; 1: enable  

执行后在当前目录下生成om模型文件：se_resnet50_fp16_bs32.om。

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用ImageNet的5万张验证集进行测试，图片与标签分别存放在/home/HwHiAiUser/dataset/ImageNet/val_union路径与/home/HwHiAiUser/dataset/ImageNet/val_label.txt文件下。  

数据集获取请参考[pytorch原始仓](https://github.com/pytorch/examples/tree/master/imagenet)说明。

### 4.2 数据集预处理

1.预处理工具为：imagenet_torch_preprocess.py  
2.执行工具命令：
```
python3 ./imagenet_torch_preprocess.py /home/HwHiAiUser/dataset/ImageNet/val_union ./data/ImageNet_bin
```
命令参数分别数据集图片路径：/home/HwHiAiUser/dataset/ImageNet/val_union和处理结果bin文件保存路径：./data/ImageNet_bin。  
执行后在./data/ImageNet_bin路径下生成数据处理后的bin文件。

### 4.3 生成数据集信息文件
1.生成数据集信息文件工具为：gen_dataset_info.py。  
2.执行工具命令：  

```
python3 ./gen_dataset_info.py bin ./data/ImageNet_bin ./data/ImageNet_bin.info 224 224
```
命令参数分别为数据集文件类型：bin、文件路径：./data/ImageNet_bin、数据集信息文件：./data/ImageNet_bin.info、图片像素长：224、图片像素宽：224。  
执行后在./data路径下生成数据集信息文件：ImageNet_bin.info。

## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

-   **[性能验证](#52-性能验证)** 

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310P上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)
### 5.2 离线推理
1.设置环境变量：

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.执行推理命令：

```
./benchmark.x86_64 -model_type=vision -om_path=./se_resnet50_fp16_bs32.om -device_id=0 -batch_size=32 -input_text_path=./data/ImageNet_bin.info -input_width=256 -input_height=256 -output_binary=false -useDvpp=false
```

分辨率(input_width，input_height)要与aipp_SE_ResNet50_pth.config文件中配置(src_image_size_w，src_image_size_h)保持一致，执行后推理结果保存在./result/dumpOutput_device0路径下。  

3.精度验证：
调用vision_metric_ImageNet.py工具脚本与数据集标签val_label.txt比对，可以获得Accuracy Top5数据：

```
python3 ./vision_metric_ImageNet.py ./result/dumpOutput_device0/ /home/HwHiAiUser/dataset/ImageNet/val_label.txt ./result accuracy_result.json
```

第一个参数为生成推理结果所在路径，第二个参数为标签数据，第三个参数为生成结果文件路径，第四个参数为生成结果文件名称。
执行后模型精度结果保存在./result/accuracy_result.json文件中

### 5.3 性能验证
1.设置环境变量：

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.执行性能测试命令：

```
./benchmark.x86_64 -round=50 -om_path=./se_resnet50_fp16_bs32.om -device_id=0 -batch_size=32  > ./result/performace_result.json
```

执行后性能测试结果保存在./result/performace_result.json文件中

## 6 评测结果

评测结果
| 模型            | pth精度                | 310P精度                   | 性能基准     | 310P性能     |
| --------------- | ---------------------- | ------------------------- | ------------ | ----------- |
| SE_ResNet50 bs32 | Acc@1 77.63,Acc@5 93.64| Acc@1 77.36,Acc@5 93.76   | 1554.726fps  | 2690.43fps  |

## 6 test目录说明

test目录下存放的为测试脚本，其中：  
1.pth2om.sh为pth模型转om模型脚本，使用命令为：

```
bash ./test/pth2om.sh /usr/local/Ascend Ascend${chip_name}
```

其中/usr/local/Ascend为cann包默认安装路径，执行后在当前目录下生成om模型: se_resnet50_fp16_bs32.om。  

2.eval_acc_perf.sh为om模型，精度、性能测试脚本，使用命令为：

```
bash ./test/eval_acc_perf.sh /usr/local/Ascend ./se_resnet50_fp16_bs32.om 32 0 /home/HwHiAiUser/dataset/ImageNet/val_label.txt
```

其中第1个参数为cann包安装路径，第2个参数为om模型，第3个参数为batch_size，第4个参数为device_id，第5个参数为标签数据。执行后精度结果保存在./result/accuracy_result.json文件中，性能结果保存在./result/performace_result.json文件中。