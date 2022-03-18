

# TextSnake Onnx模型端到端推理指导
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
[TextSnake论文](https://arxiv.org/abs/1807.01544)
论文主要提出了一种能够灵活表示任意弯曲形状文字的数据结构——TextSnake，主要思想是使用多个不同大小，带有方向的圆盘(disk)对标注文字进行覆盖，并使用FCN来预测圆盘的中心坐标，大小和方向进而预测出场景中的文字


### 1.2 代码地址

[TextSnake Pytorch实现代码](https://github.com/princewang1994/TextSnake.pytorch)

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
pytorch == 1.5.0
torchvision == 0.6.0
onnx == 1.8.0
```

### 2.2 python第三方库

```
easydict==1.8
opencv-python==4.0.0.21
scikit_image==0.14.0
numpy==1.15.1
scipy
Pillow==5.3.0
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

权重文件从百度网盘上获取：https://pan.baidu.com/s/1sU3pRBTFebbsMDac-1HsQA 密码：etdi

2.获取TextSnake源码

```shell
git clone https://github.com/princewang1994/TextSnake.pytorch
```

3.使用TextSnake_pth2onnx.py进行onnx的转换，在目录下生成TextSnake.onnx

```
python TextSnake_pth2onnx.py --input_file './textsnake_vgg_180.pth'  --output_file './TextSnake.onnx'
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
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)，需要指定输出节点以去除无用输出，可以使用netron开源可视化工具查看具体的输出节点名：

```shell
atc --model=TextSnake.onnx --framework=5 --output=TextSnake_bs1 --input_format=NCHW --input_shape="image:1,3,512,512" --log=info --soc_version=Ascend310
```



## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用Total-Text-Dataset，Images文件夹和Groundtruth文件夹根据[源码仓](https://github.com/princewang1994/TextSnake.pytorch/blob/master/dataset/total_text/download.sh)的方式获取。

新建data文件夹，进入data文件夹，创建total-text文件夹，将Images/Test移动到total-text中，将Groundtruth/Polygon/Test移动到total-text中。

### 4.2 数据集预处理
1.预处理脚本TextSnake_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件

```shell
python TextSnake_preprocess.py --src_path ./data/total-text/Images/Test --save_path ./total-text-bin
```

### 4.3 生成预处理数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```shell
python get_info.py bin ./total-text-bin ./textsnake_prep_bin.info 512 512
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
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=TextSnake_bs1.om -input_text_path=./textsnake_prep_bin.info -input_width=512 -input_height=512 -output_binary=False -useDvpp=False
```
## 6 评测结果

-   **[离线推理精度](#61-离线推理精度)**   
-   **[精度对比](#62-精度对比)**  

### 6.1 离线推理精度


调用TextSnake_postprocess.py：
```shell
python TextSnake_postprocess.py first
```
最后一个参数为本次测试的名字，执行完后会打印出精度：

```
Config: tr: 0.7 - tp: 0.6
Precision = 0.6491 - Recall = 0.5445 - Fscore = 0.5922

Config: tr: 0.8 - tp: 0.4
Precision = 0.8776 - Recall = 0.7122 - Fscore = 0.7863
```
### 6.2 精度对比
[官网精度](https://github.com/princewang1994/TextSnake.pytorch/blob/master/README.md)

tr=0.7 / tp=0.6时，Precision=0.652，Recall=0.549，F1-score=0.596

tr=0.8 / tp=0.4时，Precision=0.874，Recall=0.711，F1-score=0.784

比较离线推理精读可知，精度下降在1个点之内，因此可视为精度达标



## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[T4性能数据](#72-T4性能数据)** 
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据

batch1的性能：
 测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务

```
./benchmark.x86_64 -round=20 -om_path=TextSnake_bs1.om -device_id=0 -batch_size=1
```
执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果
```
[INFO] Dataset number: 19 finished cost 27.612ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_TextSnake_bs1_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 36.0774samples/s, ave_latency: 27.7591ms
----------------------------------------------------------------
```
Interface throughputRate: 36.0774 * 4 = 144.3096 即是batch1 310单卡吞吐率

### 7.2 T4性能数据

在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务

batch1性能

```
trtexec --onnx=TextSnake.onnx --fp16 --shapes=image:1x3x512x512
```

gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch

```
[10/09/2021-02:29:51] [I] GPU Compute
[10/09/2021-02:29:51] [I] min: 9.02429 ms
[10/09/2021-02:29:51] [I] max: 12.1098 ms
[10/09/2021-02:29:51] [I] mean: 9.35521 ms
[10/09/2021-02:29:51] [I] median: 9.2533 ms
[10/09/2021-02:29:51] [I] percentile: 11.4744 ms at 99%
[10/09/2021-02:29:51] [I] total compute time: 3.02173 s
```

batch1 t4单卡吞吐率：1000/(9.35521/1)=106.8923fps

### 7.3 性能对比

batch1：144.3096fps > 106.8923fps

310单个device的吞吐率乘4即单卡吞吐率比T4单卡的吞吐率大，故310性能高于T4性能，性能达标。
