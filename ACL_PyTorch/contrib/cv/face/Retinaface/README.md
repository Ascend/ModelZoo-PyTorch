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
	- [5 离线推理](#5-离线推理)
		- [5.1 ais\_bench工具概述](#51-ais_bench工具概述)
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
2.使用atc工具将onnx模型转换为om模型文件，工具使用方法可以参考[CANN 5.0.1 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100191944)

${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)
   
```
atc --framework 5 --model retinaface.onnx --input_shape "image:16,3,1000,1000" --soc_version Ascend${chip_name} --output retinaface_bs16 --log error --out_nodes "Concat_205:0;Softmax_206:0;Concat_155:0" --enable_small_channel 1 --insert_op_conf ./aipp.cfg
```

**注意：** `--out_nodes`参数指定onnx模型的输出节点，但不同版本的 torch包和 torchvision包可能会影响pth转换得到的 onnx 模型结构。
所以需要通过Netron网络结构可视化工具打开onnx模型，查看模型最后的`output2, output1, output0`三个节点对应的上一层节点名称确认该参数。

如果使用 torch 1.8.0 和 torchvision 0.8.0 则可以使用上面的命令，否则需要调整命令参数。

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[Wider Face官网](http://shuoyang1213.me/WIDERFACE/index.html)
的3226张验证集进行测试
### 4.2 数据集预处理
1. 确认数据集路径  
通用的数据集统一放在项目文件夹/root/datasets或/opt/npu/  
本模型数据集放在/data/widerface/val/images，可以复制数据集到该目录或者创建软链接
```
ln -s /opt/npu/wider_face/WIDER_val/images/ data/widerface/val/images
```

2. 执行预处理脚本 retinaface_pth_preprocess.py，生成数据集预处理后的bin文件
```
python3.7 retinaface_pth_preprocess.py
```
处理后的二进制文件默认放在 ./widerface 目录下

## 5 离线推理

-   **[ais_bench工具概述](#51-ais_bench工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 ais_bench工具概述

ais_bench工具为华为自研的模型推理工具。支持om模型的离线推理，能够迅速统计出模型在Ascend310P3上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程.

请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
```
mkdir result
python3.7 -m ais_bench --model retinaface_bs16.om --device 0 --batchsize 16 --input ./widerface/prep/ --output ./result --outfmt BIN
```
输出结果默认保存在 ./result/{timestamp} 下，{timestamp} 表示 ais_bench 工具执行推理任务时的时间戳。

目录下保存了每个输入对应的推理结果和性能测试文件 summary.json 文件。

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

1. 推理结果后处理

首先，将推理结果中的性能测试结果 summary.json 文件移动到其他目录
```
mv ./result/{timestamp}/sumary.json ./result/
```
然后，调用 retinaface_pth_postprocess.py 脚本处理推理结果，转换为可读结果文件
```
python3.7 retinaface_pth_postprocess.py --prediction-folder ./result/{timestamp} --info-folder ./widerface/prep_info --output-folder ./widerface_result
```
需要将其中的 {timestamp} 换成具体的推理结果路径。

参数说明：

- `--prediction-folder`：ais_bench工具的推理结果，默认为 ./result/{timestamp}
- `--info-folder`：验证集预处理时生成的info信息，默认为 ./widerface/prep_info
- `--output-folder`：处理结果的保存位置，默认为 ./widerface_result
- `--confidence-threshold`：置信度阈值，默认为 0.02

处理后的结果默认保存在 ./widerface_result 目录下

2. 计算输出结果精度：

如果是第一次运行精度计算需要运行第二步，编译评估文件，之后运行可直接执行第三步中的精度计算
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
mobildenet0.25      90.44     87.57     72.44
```
## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[T4性能数据](#72-T4性能数据)**  
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据
1. ais_bench 工具在整个数据集上推理获得性能数据  

batch1的性能：ais_bench 工具推理后输出的性能数据 
```
[INFO] --------------------Performance Summary--------------------
[INFO] H2D_latency (ms): min = 0.5559921264648438, max = 37.662506103515625, mean = 0.7194576245690456, median = 0.7016658782958984, percentile(99%) = 1.0185837745666504
[INFO] NPU_compute_time (ms): min = 0.9319999814033508, max = 1.7660000324249268, mean = 0.9820722253774902, median = 0.9710000157356262, percentile(99%) = 1.15799999923706055
[INFO] D2H_latency (ms): min = 0.7348060607910156, max = 20.054101943969727, mean = 0.9609072827731167, median = 17.602324485778812, percentile(99%) = 1.1380910873413086
[INFO] throughput 1000*batchsize(1)/NPU_compute_time.mean(0.9820722253774902): 1018.2550469906821
```
throughput: 1018.255 既为 batch1 下 Ascend310P3单卡的吞吐率

batch16的性能：ais_bench 工具推理后输出的性能数据  
```
[INFO] --------------------Performance Summary--------------------
[INFO] H2D_latency (ms): min = 8.71133804321289, max = 26.23581886291504, mean = 11.305984884205431, median = 11.35861873626709, percentile(99%) = 13.030257225036623
[INFO] NPU_compute_time (ms): min = 10.550000190734863, max = 11.541000366210938, mean = 10.64112376694632, median = 10.59749984741211, percentile(99%) = 11.324420213699343
[INFO] D2H_latency (ms): min = 11.521100997924805, max = 84.31458473205566, mean = 19.492230793037038, median = 17.60232448577881, percentile(99%) = 32.75656223297119
[INFO] throughput 1000*batchsize(16)/NPU_compute_time.mean(10.64112376694632): 1503.6005924204671
```
throughput: 1503.600 既为 batch16 下 Ascend310P3单卡的吞吐率

### 7.2 T4性能数据
在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2

batch1性能：
```
trtexec --onnx=Retinface.onnx --fp16 --shapes=image:1x3x1000x1000 --threads
```
gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch。 其中--fp16是算子精度，目前算子精度只测--fp16的。
注意--shapes是onnx的输入节点名与shape，当onnx输入节点的batch为-1时，可以用同一个onnx文件测不同batch的性能，否则用固定batch的onnx测不同batch的性能不准。

```
[07/01/2021-07:32:09] [I] GPU Compute
[07/01/2021-07:32:09] [I] min: 1.65186 ms
[07/01/2021-07:32:09] [I] max: 2.72179 ms
[07/01/2021-07:32:09] [I] mean: 1.8435 ms
[07/01/2021-07:32:09] [I] median: 1.81799 ms
[07/01/2021-07:32:09] [I] percentile: 2.70033 ms at 99%
[07/01/2021-07:32:09] [I] total compute time: 2.50163 s
```
batch1 T4单卡吞吐率：1000 * batchsize(1) / mean(1.8435) = 542.446fps  

batch16 性能：
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
batch16 T4单卡吞吐率：1000 * batchsize(16) / mean(22.174) = 721.566fps  

### 7.3 性能对比
batch1：1018.255 > 542.446

batch16：1503.600 > 721.566

可以看出，Ascend 310P3单卡的吞吐率相比于 Nvidia T4单卡的吞吐量要高出很多，故 Ascend 310P3 性能优于 Nvidia T4。

对于batch1与batch16，310P3 性能均高于 T4 性能1.2倍。

**性能优化：**  
进行softmax维度转换后性能更低，autotune对性能优化不明显，经分析无reshape，bilinear插值问题。
 


