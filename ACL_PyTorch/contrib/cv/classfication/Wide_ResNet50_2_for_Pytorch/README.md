# Wide_ResNet50_2 ONNX模型端到端推理指导

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
-   [7 性能对比](#7-性能对比)


## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[Wide_ResNet50_2论文](https://arxiv.org/abs/1605.07146)  

### 1.2 代码地址
[Wide_Resnet50_2代码地址](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
```
branch=master
commit_id=7d955df73fe0e9b47f7d6c77c699324b256fc41f
```
 
## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  
### 2.1 深度学习框架
```
CANN 5.1.RC1   
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


### 3.1 pth转onnx模型

1. 下载pth权重文件  

[wide_resnet50_2权重文件下载](https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth)

```
wget https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth
```

2. 下载模型代码
```
git clone https://github.com/pytorch/vision
cd vision
git reset 7d955df73fe0e9b47f7d6c77c699324b256fc41f --hard
python3.7 setup.py install
cd ..
```

3. 执行pth2onnx脚本，生成onnx模型文件
```python
python3.7 pth2onnx.py ./wide_resnet50_2-95faca4d.pth ./wide_resnet50_2.onnx
```
./wide_resnet50_2-95faca4d.pth为输入权重文件路径，./wide_resnet50_2.onnx
为输出onnx文件路径。运行成功后，在当前目录生成wide_resnet50_2.onnx模型文件
	
 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明

### 3.2 onnx转om模型

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考<<CANN 5.0.1 开发辅助工具指南 (推理) 01>>
使用二进制输入时，执行如下命令
```
atc --framework=5 --model=wide_resnet50_2.onnx --output=wide_resnet50_2_bs4 --input_format=NCHW --input_shape="image:4,3,224,224" --log=error --soc_version=Ascend${chip_name}
```   
参数说明：   
--model：为ONNX模型文件。   
--framework：5代表ONNX模型。   
--output：输出的OM模型。   
--input_format：输入数据的格式。   
--input_shape：输入数据的shape。   
--soc_version:处理器版本    

${chip_name}可通过`npu-smi info`指令查看，例：310P3   

![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)   


## 4 数据集预处理

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/opt/npu/imagenet/val与/opt/npu/imagenet/val_label.txt

### 4.2 数据集预处理

1.执行预处理脚本preprocess.py，生成数据集预处理后的bin文件

```
python3.7 preprocess.py /opt/npu/ImageNet/ILSVRC2012_img_val ./prep_bin
```
### 4.3 生成数据集信息文件

1.执行生成数据集信息脚本get_info.py，生成数据集信息文件

二进制输入info文件生成
```
python3.7 get_info.py bin ./prep_bin ./wide_resnet50_2_prep_bin.info 224 224
```
参数说明：   
bin为模型输入的类型，   
./prep_bin为生成的bin文件路径，   
./wide_resnet50_2_prep_bin.info为输出的info文件，   
224 224为宽高信息   


图片输入info文件生成
```
python3.7 get_info.py jpg ../dataset/ImageNet/ILSVRC2012_img_val ./ImageNet.info
```
参数说明：   
jpg为模型输入的类型，   
../dataset/ImageNet/ILSVRC2012_img_val为预处理后的数据文件相对路径，   
./ImageNet.info为输出的info文件   


## 5 离线推理
### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)   
### 5.2 离线推理(310+310P)   

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.增加benchmark可执行权限

```
chmod +x benchmark.x86_64
```

3. 执行离线推理

二进制类型输入推理命令
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=4 -om_path=wide_resnet50_2_bs4.om -input_text_path=./wide_resnet50_2_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```


## 6 离线推理精度统计

后处理统计TopN精度

调用vision_metric_ImageNet.py脚本推理结果与label比对，可以获得Accuracy Top1&Top5数据，结果保存在result.json中。
```
python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ ./val_label.txt ./ result.json
```
参数说明：   
result/dumpOutput_device0/为benchmark输出目录，   
./val_label.txt为数据集配套标签，   
./是生成文件的保存目录，   
result.json是生成的文件名。  

精度对比：   
|      | top1   | top5   |
|------|--------|--------|
| 310  | 78.48% | 94.09% |
| 310P | 78.48% | 94.09% |

## 7 性能对比
|         | 310     | 310P    | aoe后310P | T4      | 310P（aoe）/310 | 310P(aoe)/T4 |
|---------|---------|---------|----------|---------|---------------|--------------|
| bs1     | 830.52  | 731.72  |          | 487.12  |               |              |
| bs4     | 1065.75 | 1674.09 | 1729,43  | 783.15  | 1.62273       | 2.20829      |
| bs8     | 1066.03 | 1655.68 |          | 893.28  |               |              |
| bs16    | 1162.01 | 1580.17 |          | 986.95  |               |              |
| bs32    | 967.48  | 1592.23 |          | 1046.79 |               |              |
| bs64    | 802.78  | 1042.48 |          | 1076.90 |               |              |
|         |         |         |          |         |               |              |
| 最优batch | 162.01  | 1674.09 | 1729.43  | 1076.90 | 1.48830       | 1.60592      |