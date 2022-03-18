# YOLOR Onnx模型端到端推理指导
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
	-   [6.1 离线推理精度统计](#61-离线推理精度统计)
	-   [6.2 精度对比](#62-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)
	-   [7.2 T4性能数据](#72-T4性能数据)
	-   [7.3 性能对比](#73-性能对比)

## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[YOLOR论文](https://arxiv.org/abs/2105.04206)


### 1.2 代码地址

[YOLOR实现代码](https://github.com/WongKinYiu/yolor)

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
pytorch == 1.7.0
torchvision == 0.8.1
onnx == 1.7.0
```

### 2.2 python第三方库

```
Cython==0.29.24
matplotlib==3.4.3
numpy==1.21.4
opencv-python==4.5.4.58
Pillow==8.4.0
PyYAML==6.0
scipy==1.7.2
tensorboard==2.7.0
tqdm==4.62.3
seaborn==0.11.2
thop==0.0.31.post2005241907  # FLOPS computation
pycocotools==2.0.2  # COCO mAP
onnx-simplifier==0.3.6
```

安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装

```
pip install -r requirements.txt  
```



## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  


### 3.1 pth转onnx模型

1.获取pth权重文件  

权重文件从github中提供的链接中获取

2.获取yolor源码

```shell
git clone https://github.com/WongKinYiu/yolor
```

在npu上推理需要修改模型代码，把代码移植到开源模型代码中：

```
cd yolor
git am --signoff < ../yolor.patch
cd ..
```

3.使用yolor_pth2onnx.py进行onnx的转换，在目录下生成yolor_bs1.onnx

```
python yolor_pth2onnx.py --cfg ./yolor_p6_swish.cfg --weights ./yolor_p6.pt --output_file ./yolor_bs1.onnx
```

要生成其他batchsize大小的，在后面加上--batch_size参数，如：

```
python yolor_pth2onnx.py --cfg ./yolor_p6_swish.cfg --weights ./yolor_p6.pt --output_file ./yolor_bs4.onnx --batch_size 4
```

4.使用onnxsim，生成onnx_sim模型文件

```
python -m onnxsim --input-shape='1,3,1344,1344' yolor_bs1.onnx yolor_bs1_sim.onnx
```



### 3.2 onnx转om模型

1.设置环境变量

```
source env.sh
```

或

```shell
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)，需要指定输出节点以去除无用输出，可以使用netron开源可视化工具查看具体的输出节点名。

这里只保留模型的output(type: float32[1,112455,85])一个输出，其前一个算子为Concat_2575：

```shell
atc --model=yolor_bs1_sim.onnx --framework=5 --output=yolor_bs1 --input_format=NCHW --input_shape="image:1,3,1344,1344" --log=info --soc_version=Ascend310 --out_nodes="Concat_2575:0" --buffer_optimize=off_optimize
```



## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用coco数据集，Label文件根据github提供链接下载并解压得到coco文件夹，**coco文件夹需在上一级目录**。图像github提供链接下载并解压得到val2017文件夹，**并将其移动至coco/images文件夹下**。

### 4.2 数据集预处理
1.预处理脚本yolor_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件

```shell
python yolor_preprocess.py --save_path ./val2017_bin --data ./coco.yaml
```

### 4.3 生成预处理数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```shell
python get_info.py bin ./val2017_bin ./yolor_prep_bin.info 1344 1344
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息




## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)

### 5.2 离线推理
1.设置环境变量

```
source env.sh
```

或

```shell
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/
```
2.将benchmark.x86_64放到当前目录下，执行离线推理，执行时使npu-smi info查看设备状态，确保device空闲

```shell
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=yolor_bs1.om -input_text_path=./yolor_prep_bin.info -input_width=1344 -input_height=1344 -output_binary=true -useDvpp=False
```
## 6 评测结果

-   **[离线推理精度](#61-离线推理精度)**   
-   **[精度对比](#62-精度对比)**  

### 6.1 离线推理精度


调用yolor_postprocess.py：
```shell
python yolor_postprocess.py --data ./coco.yaml --img 1280 --batch 1 --conf 0.001 --iou 0.65 --npu 0 --name yolor_p6_val --names ./yolor/data/coco.names
```
最后一个参数为本次测试的名字，执行完后会打印出精度：

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.526
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.709
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.577
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.370
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.571
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.659
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.655
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.714
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.754
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.839
```
### 6.2 精度对比
[官网精度](https://github.com/WongKinYiu/yolor/blob/main/README.md)

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.52510
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.70718
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.57520
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.37058
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.56878
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.66102
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.39181
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.65229
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.71441
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.57755
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.75337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.84013
```

比较离线推理精读可知，精度下降在1个点之内，因此可视为精度达标



## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[T4性能数据](#72-T4性能数据)** 
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据

batch1的性能：
 测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务

```
./benchmark.x86_64 -round=20 -om_path=yolor_bs1.om -device_id=0 -batch_size=1
```
执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果
```
[INFO] Dataset number: 19 finished cost 597.914ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_yolor_bs1_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 1.67129samples/s, ave_latency: 599.028ms
----------------------------------------------------------------
```
Interface throughputRate: 1.6713 * 4 = 6.6852 即是batch1 310单卡吞吐率

### 7.2 T4性能数据

在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务

batch1性能

```
python onnx_infer.py yolor_bs1_sim.onnx
```

batch1 t4单卡吞吐率：6.24fps

### 7.3 性能对比

batch1：6.6852fps > 6.24fps

310单个device的吞吐率乘4即单卡吞吐率比T4单卡的吞吐率大，故310性能高于T4性能，性能达标。

