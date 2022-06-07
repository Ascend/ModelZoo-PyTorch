# DnCNN ONNX模型端到端推理指导
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
[DnCNN论文](https://ieeexplore.ieee.org/document/7839189)  

### 1.2 代码地址

brach:master

commit_id: 6b0804951484eadb7f1ea24e8e5c9ede9bea485b

备注：commitid指的是值模型基于此版本代码做的推理

[DnCNN代码](https://github.com/SaoYan/DnCNN-PyTorch)  

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```  
CANN 5.0.1
torch==1.8.0
torchvision==0.9.0
onnx==1.9.0
```

### 2.2 python第三方库

```
numpy==1.20.2
opencv-python==4.5.2.52
scikit-image==0.16.2
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)** 

-   **[onnx转om模型](#32-onnx转om模型)** 

### 3.1 pth转onnx模型

1.DnCNN模型代码下载
```
git clone https://github.com/SaoYan/DnCNN-PyTorch
cd DnCNN-PyTorch
```  
2.获取源码pth权重文件   
wget https://ascend-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/image_classification/DnCnn/net.pth  
文件的MD5sum值是： 5703a29b082cc03401fa9d9fee12cb71  

3.获取NPU训练pth文件，将net.pth文件移动到DnCNN目录下

4.编写pth2onnx脚本DnCNN_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

5.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 DnCNN_pth2onnx.py net.pth DnCNN-S-15.onnx
```

 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明

### 3.2 onnx转om模型

1.设置环境变量
```
source /usr/local/Ascend/ascend-lastest/set_env.sh
```
2.增加benchmark.{arch}可执行权限。
```
chmod u+x benchmark.x86_64
```
3.使用atc将onnx模型转换为om模型文件
（310）
```
atc --framework=5 --model=./DnCNN-S-15.onnx --input_format=NCHW --input_shape="actual_input_1:1,1,481,481" --output=DnCNN-S-15_bs1 --log=debug --soc_version=Ascend310
```
(710) for循环分别执行bs1和bs16
```
for i in 1 16;do
atc --framework=5 --model=./DnCNN-S-15.onnx --input_format=NCHW --input_shape="actual_input_1:"$i",1,481,481" --output=DnCNN-S-15_bs"$i" --log=debug --soc_version=Ascend710
done
```


## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 推理数据集获取
存放路径为 https://github.com/SaoYan/DnCNN-PyTorch 的data目录

### 4.2 数据集预处理
1.预处理脚本data_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件

```
python3.7 data_preprocess.py data ISource INoisy
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 get_info.py bin INoisy DnCNN_bin.info 481 481
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程
### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-lastest/set_env.sh
```
2.执行离线推理
for循环分别执行bs1和bs16
```
for i in 1 16;do
./benchmark.x86_64 -model_type=vision -om_path=DnCNN-S-15_bs"$i".om -device_id=0 -batch_size="$i" -input_text_path=DnCNN_bin.info -input_width=481 -input_height=481 -useDvpp=false -output_binary=true
done
```
输出结果默认保存在当前目录result/dumpOutput_deviceX(X为对应的device_id)，每个输入对应的输出对应一个_X.bin文件。

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

后处理统计TopN精度

调用postprocess.py脚本推理结果进行PSRN计算，结果会打印在屏幕上
```
python3.7 postprocess.py result/dumpOutput_device0/
```
第一个参数为benchmark输出目录
查看输出结果：
```
ISource/test064.bin PSNR 29.799832
infering...
ISource/test065.bin PSNR 31.486418
infering...
ISource/test066.bin PSNR 35.676752
infering...
ISource/test067.bin PSNR 28.577475
infering...
ISource/test068.bin PSNR 29.709767

PSNR on test data 31.526892
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上

### 6.2 开源PSNR精度
```
| Noise Level | DnCNN-S | DnCNN-B | DnCNN-S-PyTorch | DnCNN-B-PyTorch |
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|
|     15      |  31.73  |  31.61  |      31.71      |      31.60      |
|     25      |  29.23  |  29.16  |      29.21      |      29.15      |
|     50      |  26.23  |  26.23  |      26.22      |      26.20      |
```
### 6.3 精度对比
将得到的om离线模型推理PSNR值与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  

>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device。为快速获取性能数据，也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准。这里给出两种方式，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：
```
[e2e] throughputRate: 15.0465, latency: 4519.32
[data read] throughputRate: 966.417, moduleLatency: 1.03475
[preprocess] throughputRate: 525.539, moduleLatency: 1.90281
[infer] throughputRate: 22.6328, Interface throughputRate: 23.7919, moduleLatency: 43.8903
[post] throughputRate: 22.615, moduleLatency: 44.2185
```
Interface throughputRate: 23.7919，23.7919x4=95.176既是batch1 310单卡吞吐率  

batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：
```
[e2e] throughputRate: 15.3818, latency: 4420.81
[data read] throughputRate: 1484.65, moduleLatency: 0.673559
[preprocess] throughputRate: 316.273, moduleLatency: 3.16182
[infer] throughputRate: 21.4529, Interface throughputRate: 22.2853, moduleLatency: 45.6179
[post] throughputRate: 1.56798, moduleLatency: 637.764
```
Interface throughputRate: 22.2853，22.2853x4=89.1412既是batch16 310单卡吞吐率  

batch4性能：
```
[e2e] throughputRate: 15.5641, latency: 4369.02
[data read] throughputRate: 1898.17, moduleLatency: 0.526824
[preprocess] throughputRate: 523.883, moduleLatency: 1.90882
[infer] throughputRate: 22.091, Interface throughputRate: 23.9045, moduleLatency: 44.5192
[post] throughputRate: 5.50981, moduleLatency: 181.495
```
batch4 310单卡吞吐率 23.9045x4=95.618

batch8性能：
```
[e2e] throughputRate: 15.5035, latency: 4386.1
[data read] throughputRate: 1863.93, moduleLatency: 0.5365
[preprocess] throughputRate: 461.471, moduleLatency: 2.16699
[infer] throughputRate: 20.7804, Interface throughputRate: 22.2652, moduleLatency: 47.2831
[post] throughputRate: 2.74035, moduleLatency: 364.917
```
batch8 310单卡吞吐率 22.2652x4=89.0608

batch32性能：
```
[e2e] throughputRate: 12.4075, latency: 5480.54
[data read] throughputRate: 1770.65, moduleLatency: 0.564765
[preprocess] throughputRate: 242.944, moduleLatency: 4.11618
[infer] throughputRate: 15.641, Interface throughputRate: 13.2648, moduleLatency: 62.7386
[post] throughputRate: 0.68503, moduleLatency: 1459.79
```
batch32 310单卡吞吐率 13.2648x4=53.0592

**性能优化：** 

>batch32纯推理性能达标。