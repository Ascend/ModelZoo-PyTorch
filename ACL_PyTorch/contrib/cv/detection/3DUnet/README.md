# 3DUnet Onnx模型端到端推理指导
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
	-   [7.2 T4性能数据](#72-T4性能数据)
	-   [7.3 性能对比](#73-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[3DUnet论文](https://arxiv.org/abs/1606.06650)  

### 1.2 代码地址
[3DUnet代码](https://github.com/black0017/MedicalZooPytorch)  

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.0.4

pytorch >= 1.4.0
torchvision >= 0.6.0
onnx >= 1.7.0
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
[3DUnet预训练pth权重文件](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/segmentation/3DUnet/UNET3D.pth)  
```
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/segmentation/3DUnet/UNET3D.pth
```

2.安装3DUnet模型代码
```
git clone https://github.com/black0017/MedicalZooPytorch.git

```
3.编写pth2onnx脚本UNet3D_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3 UNet3D_pth2onnx.py  --input UNET3D.pth --output unet_1.onnx

```
5、使用sim简化转换的onnx文件
python3.7 -m onnxsim --input-shape="1,4,64,64,64" unet_1.onnx unet_1_sim.onnx

6、对生成的onnx进行改图脚本修改（因为resize算子不支持5HD的输入）生成act输入的onnx(第一个参数是输入的简化后的onnx文件，第二个参数是输出的改图之后的onnx文件, 第三个参数是要转的onnx对应的batchsize大小)

python3 modify.py unet_1_sim.onnx unet_1_final.onnx 1

 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明

### 3.2 onnx转om模型

1.设置环境变量
```
source /home/zl/Ascend503/ascend-toolkit/set_env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)
```
atc --framework=5 --model=./unet_1_final.onnx --output=UNet3D_bs1 --input_format=NCDHW --input_shape="image:1,4,64,64,64" --log=error --soc_version=Ascend310

```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  


### 4.1 数据集获取
该模型使用[Brats2018数据集]验证集进行测试，图片与标签分别存放在
https://ascend-pytorch-one-datasets.obs.cn-north-4.myhuaweicloud.com/train/zip/MICCAI_BraTS_2018_Data_Training.zip

### 4.2 数据集预处理
1.预处理脚本preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件(两个目录都要存在)
```
python3.7 preprocess.py --pretrained ./UNET3D.pth --output_bin ./syf_outBin1 --output_label ./syf_targetBin1/ --batchSz 1

```

## 5 离线推理

-   **[msame工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 msame工具概述
因为benchmark工具不支持3D的输入，所以采用msame的方式进行离线推理。
msame工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考(https://gitee.com/ascend/tools/tree/master/msame)
### 5.2 离线推理

2.执行离线推理
```
./main --model "./UNet3D_bs1.om" --input ./syf_outBin1 --output ./syfmsamebinout1 --outfmt BIN
```
main是msame工具生成的可执行文件。第一个输入是经过预处理脚本处理之后生成的bin文件。output参数是经过msame处理之后输出的bin文件目录（需要注意，输出目录必须是事先存在的，否则会报错）
   

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

后处理统计DSC score分数

调用postprocess.py脚本推理结果与label比对，可以获得DSC score分数
```
python3.7 postprocess.py --pretrained ./UNET3D.pth --input_bin ./syfmsamebinout1/20211217_11_36_6_921476/ --input_label ./syf_targetBin1/ --batchSz 1
```

查看输出结果：
```
--------score.avg------------ 0.25602152897045016

```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上

### 6.2 开源TopN精度
[3DUnet代码仓精度](https://github.com/black0017/MedicalZooPytorch)
```
Model        score     
3DUnet       0.2543128919787705	   
```
### 6.3 精度对比
将得到的om离线模型推理score精度比直接跑源码仓的推理脚本的精度要高，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[T4性能数据](#72-T4性能数据)**  
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据

1.msame工具在整个数据集上推理获得性能数据  
 ./main --model "./UNet3D_bs1.om" --input ./syf_outBin1 --output ./syfmsamebinout1 --outfmt BIN
batch1的性能  
```

Inference average time : 92.93 ms
Inference average time without first time: 92.92 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl



```
fps = 1 / 0.09293 = 10.75615790039798既是batch 1 310单卡的性能数据



batch16的性能
./main --model "./UNet3D_bs16.om" --input ./syf_outBin16 --output ./syfmsamebinout16 --outfmt BIN
```
Inference average time : 93.14 ms
Inference average time without first time: 93.12 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl

```
fps = 1 / 0.09314 = 10.73652566029633既是batch 16 310单卡的性能数据





batch4性能：
./main --model "./UNet3D_bs4.om" --input ./syf_outBin4 --output ./syfmsamebinout4 --outfmt BIN
```
Inference average time : 252.54 ms
Inference average time without first time: 252.53 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl


```
fps = 1 / 0.25254 = 3.959768749505029既是batch 4 310单卡的性能数据



batch8性能：
./main --model "./UNet3D_bs8.om" --input ./syf_outBin8 --output ./syfmsamebinout8 --outfmt BIN
```

Inference average time : 468.58 ms
Inference average time without first time: 468.56 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl

```
fps = 1 / 0.46858 = 2.134107302915191既是batch 8 310单卡的性能数据



batch32性能：
./main --model "./UNet3D_bs32.om" --input ./syf_outBin32 --output ./syfmsamebinout32 --outfmt BIN
```
Inference average time : 1835.80 ms
Inference average time without first time: 1836.05 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl

```
fps = 1 / 1.83580 = 0.5447216472382612既是batch 32 310单卡的性能数据

### 7.2 T4性能数据

跑的是源码推理不涉及batchsize,在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2  

```
python3 inference.py（训练生成的pth文件对应的batchsize是4）
```
gpu T4是4个device并行执行的结果，
```
----------score.avg----------- 0.2543128919787705
endtime 0.20030641555786133
fps 19.96940531764717

```




### 7.3 性能对比
batch4：3.959768749505029  <  19.96940531764717

310性能低于T4性能，性能不达标。  
该模型放在Official/cv目录下。  
 **性能优化：**  
>待优化
