# ResNet101 Onnx模型端到端推理指导
- [ResNet101 Onnx模型端到端推理指导](#resnet101-onnx模型端到端推理指导)
	- [1 模型概述](#1-模型概述)
		- [1.1 论文地址](#11-论文地址)
		- [1.2 代码地址](#12-代码地址)
	- [2 环境说明](#2-环境说明)
		- [2.1 深度学习框架](#21-深度学习框架)
		- [2.2 python第三方库](#22-python第三方库)
	- [3 模型转换](#3-模型转换)
		- [3.1 pth转onnx模型](#31-pth转onnx模型)
		- [3.2 onnx模型量化](#32-onnx模型量化)
		- [3.3 onnx转om模型](#33-onnx转om模型)
	- [4 数据集预处理](#4-数据集预处理)
		- [4.1 数据集获取](#41-数据集获取)
		- [4.2 数据集预处理](#42-数据集预处理)
		- [4.3 生成数据集信息文件](#43-生成数据集信息文件)
	- [5 离线推理](#5-离线推理)
		- [5.1 benchmark工具概述](#51-benchmark工具概述)
		- [5.2 离线推理](#52-离线推理)
	- [6 精度对比](#6-精度对比)
		- [6.1 离线推理TopN精度统计](#61-离线推理topn精度统计)
		- [6.2 开源TopN精度](#62-开源topn精度)
		- [6.3 精度对比](#63-精度对比)
	- [7 性能对比](#7-性能对比)
		- [7.1 npu性能数据](#71-npu性能数据)
		- [7.2 T4性能数据](#72-t4性能数据)
		- [7.3 性能对比](#73-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[ResNet101论文](https://arxiv.org/pdf/1512.03385.pdf)  

### 1.2 代码地址
[ResNet101代码](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)  
branch:master
commit_id:7d955df73fe0e9b47f7d6c77c699324b256fc41f

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.0.4

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
请参考[pytorch原始仓](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)给出的ResNet101权重文件下载地址获取权重文件：resnet101-63fe2227.pth

2.ResNet101模型代码在torchvision里，安装torchvision，arm下需源码安装，参考torchvision官网，若安装过程报错请百度解决

```
git clone https://github.com/pytorch/vision
cd vision
python3.7 setup.py install
cd ..
```
3.编写pth2onnx脚本resnet101_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 resnet101_pth2onnx.py ./resnet101-63fe2227.pth resnet101.onnx
```

 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明

### 3.2 onnx模型量化(可选)

1.AMCT工具包安装，具体参考《[CANN 开发辅助工具指南  01](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》中的昇腾模型压缩工具使用指南（ONNX）章节；

2.生成bin格式数据集，数据集用于校正量化因子。当前模型为动态batch，建议使用较大的batch size：

```
python3.7 gen_calibration_bin.py resnet /root/datasets/imagenet/val ./calibration_bin 32 1
```

参数说明：

- resnet：模型类型
- /root/datasets/imagenet/val：模型使用的数据集路径；
- ./calibration_bin：生成的bin格式数据集路径；
- 32：batch size；
- 1：batch num。

3.ONNX模型量化

```
amct_onnx calibration --model resnet101.onnx  --save_path ./result/resnet101  --input_shape "image:32,3,224,224" --data_dir "./calibration_bin" --data_types "float32" 
```

会在result目录下生成resnet101_deploy_model.onnx量化模型

4.量化模型后续的推理验证流程和非量化一致。

### 3.3 onnx转om模型

1. 设置环境变量

	```
	source /usr/local/Ascend/ascend-toolkit/set_env.sh
	```
2. 使用atc将onnx模型转换为om模型文件，工具使用方法可以参考《[CANN 开发辅助工具指南  01](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》中的ATC工具使用指南章节

	```
	atc --framework=5 --model=./resnet101.onnx --output=resnet101_bs16 --input_format=NCHW --input_shape="image:16,3,224,224" --log=debug --soc_version=Ascend310 --insert_op_conf=aipp.config
	```

**说明：**  

> 若设备类型为Ascend310P，设置--soc_version=Ascend${chip_name}（Ascend310P3）， ${chip_name}可通过`npu-smi info`指令查看；
> ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)
>
> aipp.config是AIPP工具数据集预处理配置文件，详细说明可参考"ATC工具使用指南"中的"AIPP配置"章节。

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用ImageNet的5万张验证集进行测试，图片与标签分别存放在/root/datasets/imagenet/val与/root/datasets/imagenet/val_label.txt。

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
python3.7 gen_dataset_info.py bin ./prep_dataset ./resnet101_prep_bin.info 256 256
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310、310P上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考《[CANN 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》
### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=resnet101_bs16.om -input_text_path=./resnet101_prep_bin.info -input_width=256 -input_height=256 -output_binary=False -useDvpp=False
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
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "77.37%"}, {"key": "Top2 accuracy", "value": "87.1%"}, {"key": "Top3 accuracy", "value": "90.61%"}, {"key": "Top4 accuracy", "value": "92.42%"}, {"key": "Top5 accuracy", "value": "93.54%"}]}
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上

### 6.2 开源TopN精度
[torchvision官网精度](https://pytorch.org/vision/stable/models.html)
```
Model         Acc@1     Acc@5
ResNet-101    77.374    93.546
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试
