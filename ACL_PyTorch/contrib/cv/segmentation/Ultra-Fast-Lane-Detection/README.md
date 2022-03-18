# Ultra-Fast-Lane-Detection推理说明

## 1 概述

### 1.1 模型论文

模型论文链接：“[Ultra Fast Structure-aware Deep Lane Detection](https://arxiv.org/abs/2004.11757)”:

### 1.2 代码链接

开源代码链接：“[Ultra-Fast-Lane-Detection代码](https://github.com/cfzd/Ultra-Fast-Lane-Detection)”



## 2 环境准备

### 2.1 环境依赖

首先设置环境：

```shell
source set_env.sh
```
依赖无明确要求，以下环境仅供参考（已经写入requirements.txt）：

```shell
torch==1.9.1
torchvision==0.10.0
numpy==1.19.5
scipy==1.6.2
onnx==1.10.2
onnxruntime==1.9.0
json5==0.9.6
scikit-learn==0.24.2 
opencv-python==4.5.3.56
```

### 2.2 获取数据集

本模型使用的是Tusimple数据集：[Tusimple数据集](https://github.com/TuSimple/tusimple-benchmark)

### 2.3 获取开源模型代码

执行命令：

```shell
git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection.git
commit_id=f77c240fb2b950a2f8a823ba1ead85b3b9ae9dbd
```

### 2.4 获取权重文件

原模型开源仓中pth可以再百度云盘中获取：
pth文件（百度云盘）：[BaiduDrive(code:bghd)](https://pan.baidu.com/s/1Fjm5yVq1JDpGjh4bdgdDLA)



### 2.5 获取benchmark工具

在当前工作目录下安装一个benchmar.x86_64工具。


## 3 模型转换

### 3.1 pth转onnx

运行脚本	pth_to_onnx.py，将已获取的pth文件转换为支持输入动态Batch Size的onnx模型(tusimple_Dynamic.onnx)。

执行命令：

```shell
python pth_to_onnx.py ${onnx path} ${pth path}
```



### 3.2 onnx转om

在当前目录下，使用华为开发的CANN包中的atc功能和ultra_aipp.config进行模型转换：

```shell
atc --framework=5 --model=./tusimple_Dynamic.onnx --output=aipp_test_bs16  --log=info --soc_version=Ascend310 --output_type=FP16 --insert_op_conf=./ultra_aipp.config  --input_shape="input:16,3,288,800"
```



## 4 数据集预处理

在已经通过链接获取的数据集进行初步处理，处理包括：
1.将所有压缩文件解压
2.将每个目录下的clips文件夹合并 
在处理完的数据集文件夹下放入脚本 	data_preprocess.py，并运行脚本。

```shell
python data_preprocess.py ${数据集路径} ${bin文件的存放路径（当前文件夹下）} ${info路径（当前文件夹下）}
```

此脚本完成的工作是：

1. 在原数据集中抽取出用于推理的 JPG文件，在本目录的Inference_images_dataProcess目录下；
2. 将抽取出来的 JPG文件的信息转换为bin文件，在本目录的Inference_bin目录下；
3. 生成用于推理的info文件，本目录下的test_dataset.info文件；



## 5 推理

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN 5.0.1 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100191895?idPath=23710424%7C251366513%7C22892968%7C251168373)

### 5.2 执行推理

在当前工作目录下使用benchmark工具进行推理

执行命令：

```shell
./benchmark.x86_64 -model_type=vision -device_id=3 -batch_size=16 -om_path=aipp_test_bs16.om -input_text_path=/opt/npu/TUSIMPLEROOT/test_dataset.info -input_width=288 -input_height=800 -output_binary=False -useDvpp=False
```



## 6 精度对比

### 6.1推理精度统计

使用ufld_postprocess.py脚本进行精度统计

执行命令：

```shell
python ufld_postprocess.py ${inference result path} ${dataset path}
```

获得精度统计结果：

```
Predict Lane Processed: 2780/2782
Predict Lane Processed: 2781/2782
Predict Lane Processed: 2782/2782
Predict File generated.
Accuracy 0.9581503971106696
```

### 6.2 开源精度

使用相同的pth文件，论文中的精达到**95.82%**。

### 6.3 精度对比

NPU推理精度达到**95.81%**，与开源精度相比相差在规定范围以内，故精度达标。

## 7 性能对比

### 7.1 NPU性能数据

在当前目录下，使用benchmark工具会给出om模型的推理性能，执行命令：

```shell
./benchmark.x86_64 -round=100 -om_path=${om path} -device_id=${device id} -batch_size=16
```

bs16 运行结果：

```
[INFO] Dataset number: 97 finished cost 49.436ms
[INFO] Dataset number: 98 finished cost 49.493ms
[INFO] Dataset number: 99 finished cost 49.343ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_aipp_no_export_bs16_in_device_2.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 323.826samples/s, ave_latency: 3.08851ms
----------------------------------------------------------------
```

bs16 310单卡吞吐率：323.826x4=1295.304fps/card  



bs1 运行结果：

```
[INFO] Dataset number: 97 finished cost 5.473ms
[INFO] Dataset number: 98 finished cost 5.471ms
[INFO] Dataset number: 99 finished cost 5.46ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_aipp_test_bs1_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 182.044samples/s, ave_latency: 5.50531ms
----------------------------------------------------------------
```

bs1 310单卡吞吐率：182.044x4=728.176fps/card

### 7.2 T4性能数据

使用 Nvidia Tesla T4对ONNX模型进行离线推理性能的统计，执行命令：

```shell
trtexec --onnx=${onnx path} --fp16 --shapes=input:16x3x288x800 --threads
```

bs16 运行结果：

```
[12/09/2021-15:37:06] [I] throughput: 0 qps
[12/09/2021-15:37:06] [I] walltime: 3.04011 s
[12/09/2021-15:37:06] [I] Enqueue Time
[12/09/2021-15:37:06] [I] min: 0.532959 ms
[12/09/2021-15:37:06] [I] max: 0.609924 ms
[12/09/2021-15:37:06] [I] median: 0.553345 ms
[12/09/2021-15:37:06] [I] GPU Compute
[12/09/2021-15:37:06] [I] min: 12.2784 ms
[12/09/2021-15:37:06] [I] max: 13.4615 ms
[12/09/2021-15:37:06] [I] mean: 12.8239 ms
[12/09/2021-15:37:06] [I] median: 12.8769 ms
[12/09/2021-15:37:06] [I] percentile: 13.2598 ms at 99%
[12/09/2021-15:37:06] [I] total compute time: 3.02645 s
```

bs16 T4上的推理性能：1000/(12.8239/16)=1247.670fps/card

bs1 运行结果：

```
[12/09/2021-15:01:31] [I] throughput: 0 qps
[12/09/2021-15:01:31] [I] walltime: 3.00416 s
[12/09/2021-15:01:31] [I] Enqueue Time
[12/09/2021-15:01:31] [I] min: 0.231201 ms
[12/09/2021-15:01:31] [I] max: 0.637817 ms
[12/09/2021-15:01:31] [I] median: 0.251221 ms
[12/09/2021-15:01:31] [I] GPU Compute
[12/09/2021-15:01:31] [I] min: 1.48444 ms
[12/09/2021-15:01:31] [I] max: 1.69983 ms
[12/09/2021-15:01:31] [I] mean: 1.52162 ms
[12/09/2021-15:01:31] [I] median: 1.51758 ms
[12/09/2021-15:01:31] [I] percentile: 1.68805 ms at 99%
[12/09/2021-15:01:31] [I] total compute time: 2.99759 s
&&&& PASSED TensorRT.trtexec # trtexec --onnx=tusimple_Dynamic.onnx --fp16 --shapes=input:1x3x288x800 --threads

```

bs1 T4上的推理性能：1000/(1.52162/1)=657.194306fps/card



### 7.3 性能对比

NPU上性能好于T4上的性能，故性能达标。

### 7.4 性能优化

使用AIPP方法对模型进行优化后，模型推理性能提高

执行命令生成优化后的om模型：

```shell
atc --framework=5 --model=./tusimple_Dynamic.onnx --output=aipp_test_bs16  --log=info --soc_version=Ascend310 --output_type=FP16 --insert_op_conf=./ultra_aipp.config  --input_shape="input:16,3,288,800"
```

