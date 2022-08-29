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
CANN 5.1.RC1
python == 3.7.5
pytorch >= 1.8.1
torchvision >= 0.8.1
onnx >= 1.9.0
```

### 2.2 python第三方库

```
numpy == 1.19.2
Pillow == 8.2.0
opencv-python == 4.5.2.54
skl2onnx == 1.8.0
scikit-learn == 0.24.1
h5py == 3.3.0
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
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.1.RC1 开发辅助工具指南 (推理) 01
${chip_name}可使用 npu-smi info 查看(对应name属性)，例：310p3

![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```
atc --framework=5 --model=./model/model_mkt1501_bs1.onnx --input_format=NCHW --input_shape="image:1,3,384,128" --output=mgn_mkt1501_bs1 --log=debug --soc_version=Ascend${chip_name}
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
source /usr/local/Ascend/ascend-toolkit/set_env.sh
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
mAP: 0.9423
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
1.benchmark工具在整个数据集上推理获得性能数据(优化在310p上产生，故对比数据采用310p的初始数据 )  

 ``` 
MGN模型	未任何优化前310p（单卡吞吐率）	优化后310p（单卡吞吐率）
bs1	      362.752 fps	                640.829 fps
bs4	      2156.73 fps	                1453.49 fps
bs8	      1281.93 fps	                1519.17 fps
bs16	  1167.81 fps	                1388.27 fps
bs32	  1096.42 fps	                1367.63 fps
bs64	  1107.77 fps	                1364.84 fps
```