# Retinface Onnx模型端到端推理指导
- [Retinface Onnx模型端到端推理指导](#retinface-onnx模型端到端推理指导)
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
		- [6.1 离线推理精度统计](#61-离线推理精度统计)
		- [6.2 开源精度](#62-开源精度)
		- [6.3 精度对比](#63-精度对比)
	- [7 性能对比](#7-性能对比)
		- [7.1 npu性能数据](#71-npu性能数据)
		- [7.2 T4性能数据](#72-t4性能数据)
		- [7.3 性能对比](#73-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[Retinface论文](https://arxiv.org/abs/1905.00641)  

### 1.2 代码地址
[Retinface代码](https://github.com/biubug6/Pytorch_Retinaface)  
branch:master  
commit id:b984b4b775b2c4dced95c1eadd195a5c7d32a60b

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.0.1

pytorch >= 1.5.0
torchvision >= 0.6.0
onnx >= 1.7.0
```

### 2.2 python第三方库

```
numpy == 1.20.3
Pillow == 8.2.0
opencv-python == 4.5.2.54
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
[Retinface预训练pth权重文件 百度云](https://pan.baidu.com/s/12h97Fy1RYuqMMIV-RpzdPg) Password: fstq

2.Retinface模型代码在github上
```
git clone https://github.com/biubug6/Pytorch_Retinaface
cd Pytorch_Retinaface
git reset b984b4b775b2c4dced95c1eadd195a5c7d32a60b --hard
```
 **说明：**  
>注意目前ATC支持的onnx算子版本为11

3.安装必要依赖，主要是tqdm，Cython，ipython为必要
```
cd .. 
pip install -r requirments.txt
```

4.执行pth2onnx脚本，生成onnx模型文件，mobilenet0.25_Final.pth为3.1节链接下载，放在当前目录下
```
python3.7 pth2onnx.py -m mobilenet0.25_Final.pth
```

### 3.2 onnx转om模型

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.使用atc工具将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01

${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)
   
```
atc --framework 5 --model retinaface.onnx --input_shape "image:16,3,1000,1000" --soc_version Ascend${chip_name} --output retinaface_bs16 --log error --out-nodes="Concat_205:0;Softmax_206:0;Concat_155:0" --enable_small_channel=1 --insert_op_conf=./aipp.cfg
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[Wider Face官网](http://shuoyang1213.me/WIDERFACE/index.html)
的3226张验证集进行测试
### 4.2 数据集预处理
数据集路径  
通用的数据集统一放在项目文件夹/root/datasets或/opt/npu/  
本模型数据集放在/data/widerface/val/images
```
ln -s /opt/npu/wider_face/WIDER_val/images/ data/widerface/val/images
```
1.预处理脚本retinaface_pth_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 retinaface_pth_preprocess.py
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本gen_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 get_info.py bin ./widerface/prep ./retinface_prep_bin.info 1000 1000
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
2.执行离线推理，确保benchmark工具在当前目录下，使用uname -m，检查本地环境架构，使用合理的benchmark工具如x86_64版本
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=retinaface_bs16.om -input_text_path=./retinface_prep_bin.info -input_width=1000 -input_height=1000 -output_binary=True -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_device{0}
## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

后处理统计精度

调用retinaface_pth_postprocess.py脚本产生推理结果。
```
python3.7 retinaface_pth_postprocess.py 
```
查看输出结果：如果第一次运行需要运行第二步，编译评估文件
```
1 cd Pytorch_Retinaface/widerface_evaluate
2 python3.7 setup.py build_ext --inplace
3 python3.7 evaluation.py -p ../../widerface_result/
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如下描述  

### 6.2 开源精度
[torchvision官网精度](https://pytorch.org/vision/stable/models.html)
```
Model               Easy     Medium    Hard
mobildenet0.25      90.70     88.16    73.82
```
### 6.3 精度对比
将得到的om离线模型推理精度与该模型github代码仓上公布的精度对比，精度下降在几乎1%范围之内，故精度达标。  
```
Model               Easy     Medium    Hard
mobildenet0.25      90.4     87.57     72.43
```
## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[T4性能数据](#72-T4性能数据)**  
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 20.3771, latency: 158315
[data read] throughputRate: 20.524, moduleLatency: 48.7234
[preprocess] throughputRate: 20.517, moduleLatency: 48.7401
[infer] throughputRate: 20.518, Interface throughputRate: 106.197, moduleLatency: 37.7644
[post] throughputRate: 20.5174, moduleLatency: 48.7391
```
Interface throughputRate: ，106.197x4=424.778既是batch1 310单卡吞吐率  
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  
```
[e2e] throughputRate: 20.7614, latency: 155385
[data read] throughputRate: 20.9727, moduleLatency: 47.6809
[preprocess] throughputRate: 20.9568, moduleLatency: 47.7173
[infer] throughputRate: 20.8837, Interface throughputRate: 136.207, moduleLatency: 35.6322
[post] throughputRate: 1.30709, moduleLatency: 765.059
```
Interface throughputRate: 544.828，136.207x4=544.828 既是batch16 310单卡吞吐率

### 7.2 T4性能数据
在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2
batch1性能：
```
trtexec --onnx=Retinface.onnx --fp16 --shapes=image:1x3x1000x1000 --threads
```
gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch。其中--fp16是算子精度，目前算子精度只测--fp16的。注意--shapes是onnx的输入节点名与shape，当onnx输入节点的batch为-1时，可以用同一个onnx文件测不同batch的性能，否则用固定batch的onnx测不同batch的性能不准  
```
[07/01/2021-07:32:09] [I] GPU Compute
[07/01/2021-07:32:09] [I] min: 1.65186 ms
[07/01/2021-07:32:09] [I] max: 2.72179 ms
[07/01/2021-07:32:09] [I] mean: 1.8435 ms
[07/01/2021-07:32:09] [I] median: 1.81799 ms
[07/01/2021-07:32:09] [I] percentile: 2.70033 ms at 99%
[07/01/2021-07:32:09] [I] total compute time: 2.50163 s
```
batch1 t4单卡吞吐率：542.446fps  

batch16性能：
```
trtexec --onnx=Retinface.onnx --fp16 --shapes=image:16x3x1000x1000 --threads
```
```
[07/01/2021-07:37:48] [I] GPU Compute
[07/01/2021-07:37:48] [I] min: 21.3848 ms
[07/01/2021-07:37:48] [I] max: 22.6833 ms
[07/01/2021-07:37:48] [I] mean: 22.174 ms
[07/01/2021-07:37:48] [I] median: 22.1829 ms
[07/01/2021-07:37:48] [I] percentile: 22.6833 ms at 99%
[07/01/2021-07:37:48] [I] total compute time: 2.15088 s
```
batch32 t4单卡吞吐率：717.328fps  
### 7.3 性能对比
batch1：424.778 < 542.446 
batch16：544.828 < 721.566
310单个device的吞吐率乘4即单卡吞吐率比T4单卡的吞吐率小，故310性能低于T4性能。
对于batch1与batch16，310性能均高于T4性能1.2倍，该模型放在Benchmark/cv/classification目录下。
batch1 * 1.2 < batch 16 ,
**性能优化：**  
进行softmax维度转换后性能更低，autotune对性能优化不明显，经分析无reshape，bilinear插值问题
 


