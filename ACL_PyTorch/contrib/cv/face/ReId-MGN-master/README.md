# MGN Onnx模型端到端推理指导
- [MGN Onnx模型端到端推理指导](#MGN-onnx模型端到端推理指导)
	- [1 模型概述](#1-模型概述)
		- [1.1 论文地址](#11-论文地址)
		- [1.2 代码地址](#12-代码地址)
	- [2 环境说明](#2-环境说明)
		- [2.1 深度学习框架](#21-深度学习框架)
		- [2.2 python第三方库](#22-python第三方库)
	- [3 模型转换](#3-模型转换)
		- [3.1 pth转onnx模型](#31-pth转onnx模型)
		- [3.2 onnx转om模型](#32-onnx转om模型)
	- [4 数据集预处理](#4-数据集预处理)
		- [4.1 数据集获取](#41-数据集获取)
		- [4.2 数据集预处理](#42-数据集预处理)
		- [4.3 生成数据集信息文件](#43-生成数据集信息文件)
	- [5 离线推理](#5-离线推理)
		- [5.1 benchmark工具概述](#51-benchmark工具概述)
		- [5.2 离线推理](#52-离线推理)
	- [6 精度对比](#6-精度对比)
		- [6.1 离线推理mAP精度](#61-离线推理mAP精度)
		- [6.2 开源mAP精度](#62-开源mAP精度)
		- [6.3 精度对比](#63-精度对比)
	- [7 性能对比](#7-性能对比)
		- [7.1 npu性能数据](#71-npu性能数据)


## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[MGN论文](https://arxiv.org/pdf/1804.01438.pdf)  

### 1.2 代码地址
[MGN代码](https://github.com/GNAYUOHZ/ReID-MGN)  

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.0.1
python = 3.7.5
pytorch >= 1.5.0
torchvision >= 0.6.0
onnx >= 1.7.0
```

### 2.2 python第三方库

```
numpy == 1.20.3
Pillow == 8.2.0
opencv-python == 4.5.2.54
albumentations == 0.5.2
```
**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.MGN模型代码下载
```
cd ./ReId-MGN-master
git clone https://github.com/GNAYUOHZ/ReID-MGN.git ./MGN
patch -R MGN/data.py < module.patch
```
2.预训练模型获取。
```
到以下链接下载预训练模型，并放在/model目录下：
(https://pan.baidu.com/s/12AkumLX10hLx9vh_SQwdyw) password:mrl5
```

3.编写pth2onnx脚本pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
#将model.pt模型转为market1501.onnx模型，注意，生成onnx模型名(第二个参数)和batch size(第三个参数)根据实际大小设置.
python3.7 ./pth2onnx.py ./model/model.pt ./model/model_mkt1501_bs1.onnx 1        
```

 **模型转换要点：**  
### 3.2 onnx转om模型

1.设置环境变量
```
export install_path=/usr/local/Ascend/ascend-toolkit/latest

export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH

export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH

export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH

export ASCEND_OPP_PATH=${install_path}/opp
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01
```
atc --framework=5 --model=./model/model_mkt1501.onnx --input_format=NCHW --input_shape="image:1,3,284,128" --output=mgn_mkt1501_bs1 --log=debug --soc_version=Ascend310
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型将[Market1501数据集](https://pan.baidu.com/s/1ntIi2Op?_at_=1624593258681) 的训练集随机划分为训练集和验证集，为复现精度这里采用固定的验证集。

### 4.2 数据集预处理
1.将下载好的数据集移动到./ReID-MGN-master/data目录下

2.执行预处理脚本，生成数据集预处理后的bin文件
```
# 首先在要cd到ReID-MGN-master目录下.
python3  ./postprocess_MGN.py --mode save_bin  --data_path ./data/market1501
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本preprocess_MGN.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python ./preprocess_MGN.py bin ./data/market1501/bin_data/q/ ./q_bin.info 384 128
python ./preprocess_MGN.py bin ./data/market1501/bin_data/g/ ./g_bin.info 384 128

python ./preprocess_MGN.py bin ./data/market1501/bin_data_flip/q/ ./q_bin_flip.info 384 128
python ./preprocess_MGN.py bin ./data/market1501/bin_data_flip/g/ ./g_bin_flip.info 384 128
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息  
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理benchmark工具用户指南 01
### 5.2 离线推理
1.设置环境变量
```
export install_path=/usr/local/Ascend/ascend-toolkit/latest

export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH

export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH

export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH

export ASCEND_OPP_PATH=${install_path}/opp
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=mgn_mkt1501_bs1.om -input_text_path=./q_bin.info -input_width=384 -input_height=128 -output_binary=False -useDvpp=False
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=mgn_mkt1501_bs1.om -input_text_path=./g_bin.info -input_width=384 -input_height=128 -output_binary=False -useDvpp=False

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=mgn_mkt1501_bs1.om -input_text_path=./q_bin_flip.info -input_width=384 -input_height=128 -output_binary=False -useDvpp=False
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=mgn_mkt1501_bs1.om -input_text_path=./g_bin_flip.info -input_width=384 -input_height=128 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_deviceX(X为对应的device_id)，每个输入对应一个_X.txt文件的输出。

## 6 精度对比

-   **[离线推理mAP精度](#61-离线推理mAP精度)**  
-   **[开源mAP精度](#62-开源mAP精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理mAP精度

后处理统计mAP精度

调用postprocess_MGN.py脚本的“evaluate_om”模式推理结果与语义分割真值进行比对，可以获得mAP精度数据。
```
python3.7 ./postprocess_MGN.py  --mode evaluate_om --data_path ./data/market1501/ 
```
第一个参数为main函数运行模式，第二个为原始数据目录，第三个为模型所在目录。  
查看输出结果：
```
mAP: 0.9433
```
经过对bs8的om测试，本模型batch8的精度没有差别，精度数据均如上。

### 6.2 开源mAP精度
[原代码仓公布精度](https://github.com/GNAYUOHZ/ReID-MGN/README.md)
```
Model       mAP  
MGN         0.9433  
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上。

### 6.3 精度对比
将得到的om离线模型推理mAP精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
1.benchmark工具在整个数据集上推理获得性能数据  

batch1初始性能：
```
[e2e] throughputRate: 84.5598, latency: 39829.8
[data read] throughputRate: 473.148, moduleLatency: 2.1135
[preprocess] throughputRate: 393.066, moduleLatency: 2.5441
[infer] throughputRate: 88.2882, Interface throughputRate: 111.277, moduleLatency: 10.4851
[post] throughputRate: 88.2756, moduleLatency: 11.3282
```
batch1 310单卡吞吐率：111.277 * 4 = 445.108fps

batch1优化后性能：
[e2e] throughputRate: 96.8416, latency: 34778.5
[data read] throughputRate: 446.222, moduleLatency: 2.24104
[preprocess] throughputRate: 422.735, moduleLatency: 2.36555
[infer] throughputRate: 99.7554, Interface throughputRate: 128.15, moduleLatency: 9.25689
[post] throughputRate: 99.7383, moduleLatency: 10.0262

batch1 310单卡吞吐率：128.15 * 4 = 512.6fps

batch4性能：
```
[e2e] throughputRate: 106.025, latency: 31766.1
[data read] throughputRate: 500.107, moduleLatency: 1.99957
[preprocess] throughputRate: 471.221, moduleLatency: 2.12215
[infer] throughputRate: 108.979, Interface throughputRate: 153.797, moduleLatency: 8.03481
[post] throughputRate: 27.2346, moduleLatency: 36.7181
```
batch4 310单卡吞吐率：153.797 * 4 = 615.188fps

batch8性能：
```
[e2e] throughputRate: 106.324, latency: 149665
[data read] throughputRate: 121.058, moduleLatency: 8.2605
[preprocess] throughputRate: 120.662, moduleLatency: 8.28762
[infer] throughputRate: 107.422, Interface throughputRate: 149.297, moduleLatency: 8.18964
[post] throughputRate: 13.4334, moduleLatency: 74.4414
```
batch8 310单卡吞吐率：149.297 * 4 = 597.188fps  

batch16初始性能：
```
[e2e] throughputRate: 103.095, latency: 32668.8
[data read] throughputRate: 138.066, moduleLatency: 7.24292
[preprocess] throughputRate: 135.594, moduleLatency: 7.37498
[infer] throughputRate: 107.451, Interface throughputRate: 147.867, moduleLatency: 8.19638
[post] throughputRate: 6.72704, moduleLatency: 148.654
```
batch16 310单卡吞吐率：147.867 * 4 = 591.468fps

batch16优化后性能：
```
[e2e] throughputRate: 121.183, latency: 27792.7
[data read] throughputRate: 138.209, moduleLatency: 7.23544
[preprocess] throughputRate: 135.553, moduleLatency: 7.37721
[infer] throughputRate: 125.617, Interface throughputRate: 184.74, moduleLatency: 6.86643
[post] throughputRate: 7.86374, moduleLatency: 127.166
```
batch16 310单卡吞吐率：184.74 * 4 = 738.96fps

batch32性能：
```
[e2e] throughputRate: 109.639, latency: 30719.1
[data read] throughputRate: 144.87, moduleLatency: 6.90276
[preprocess] throughputRate: 141.787, moduleLatency: 7.05281
[infer] throughputRate: 112.348, Interface throughputRate: 159.033, moduleLatency: 7.70075
[post] throughputRate: 3.53321, moduleLatency: 283.029
```
batch32 310单卡吞吐率：159.033 * 4 = 636.132fps

 ``` 
MGN模型	未任何优化前310（单卡吞吐率）	优化transdata、transpose后310（单卡吞吐率）
bs1	      445.108fps	                512.6fps
bs16	  591.468fps	                738.96fps
```