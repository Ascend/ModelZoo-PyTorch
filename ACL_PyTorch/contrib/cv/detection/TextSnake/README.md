

# TextSnake ONNX模型端到端推理指导
- [TextSnake ONNX模型端到端推理指导](#textsnake-onnx模型端到端推理指导)
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
		- [4.3 生成预处理数据集信息文件](#43-生成预处理数据集信息文件)
	- [5 离线推理](#5-离线推理)
		- [5.1 benchmark工具概述](#51-benchmark工具概述)
		- [5.2 离线推理](#52-离线推理)
	- [6 精度对比](#6-精度对比)
	- [7 性能对比](#7-性能对比)
		- [7.1 310P性能数据](#71-310p性能数据)
		- [7.2 T4性能数据](#72-t4性能数据)
		- [7.3 性能对比](#73-性能对比)

## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[TextSnake论文](https://arxiv.org/abs/1807.01544)
论文主要提出了一种能够灵活表示任意弯曲形状文字的数据结构——TextSnake，主要思想是使用多个不同大小，带有方向的圆盘(disk)对标注文字进行覆盖，并使用FCN来预测圆盘的中心坐标，大小和方向进而预测出场景中的文字

### 1.2 代码地址

[TextSnake Pytorch实现代码](https://github.com/princewang1994/TextSnake.pytorch)
```
branch=master 
commit_id=b4ee996d5a4d214ed825350d6b307dd1c31faa07
```

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN == 5.1.RC1
pytorch == 1.5.0
torchvision == 0.6.0
onnx == 1.8.0
```

### 2.2 python第三方库

```
easydict==1.8
opencv-python==4.1.2.30
scikit_image==0.14.0
numpy==1.15.1
scipy==1.5.4
Pillow==5.3.0
shapely
tqdm
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
3.使用TextSnake_pth2onnx.py进行onnx的转换，在目录下生成TextSnake.onnx，注意将网盘里的模型名称修改为textsnake_vgg_180.pth

```
python TextSnake_pth2onnx.py --input_file './textsnake_vgg_180.pth'  --output_file './TextSnake.onnx'
```



### 3.2 onnx转om模型

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23310P424%7C251366513%7C22892968%7C251168373)，需要指定输出节点以去除无用输出，可以使用netron开源可视化工具查看具体的输出节点名：

使用atc将onnx模型 ${chip_name}可通过npu-smi info指令查看

![输入图片说明](https://images.gitee.com/uploads/images/2022/0704/095450_881600a3_7629432.png "屏幕截图.png")

执行ATC命令

```shell
atc --model=TextSnake.onnx \
--framework=5 \
--output=TextSnake_bs1 \
--input_format=NCHW \
--input_shape="image:1,3,512,512" \
--log=info \
--soc_version=Ascend${chip_name} \
```
参数说明：\
--model：为ONNX模型文件。 \
--framework：5代表ONNX模型。\
--output：输出的OM模型。\
--input_format：输入数据的格式。\
--input_shape：输入数据的shape。\
--log：日志级别。\
--soc_version：处理器型号。\

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用Total-Text-Dataset
1.首先进入TextSnake.pytorch/dataset/total-text文件夹中
根据[源码仓](https://github.com/princewang1994/TextSnake.pytorch/tree/master/dataset/total_text/download.sh)的方式下载数据集并整理成gt文件夹和Images文件夹。

2.回到TextSnake目录下新建data文件夹，进入data文件夹，创建total-text文件夹，将第一步生成的Images/Test移动到total-text中，将gt/Test移动到total-text中。

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

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23310P424%7C251366513%7C22892968%7C251168373)

### 5.2 离线推理
1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.将benchmark.x86_64放到当前目录下，执行离线推理，执行时使npu-smi info查看设备状态，确保device空闲

```shell
./benchmark.x86_64 \
-model_type=vision \
-device_id=0 \
-batch_size=1 \
-om_path=TextSnake_bs1.om \
-input_text_path=./textsnake_prep_bin.info \ -input_width=512 \
-input_height=512 \
-output_binary=False \
-useDvpp=False \
```

## 6 精度对比  

调用TextSnake_postprocess.py：
```shell
python TextSnake_postprocess.py first
```
最后一个参数为"first"为生成结果文件的路径。
执行完后会打印出精度：

```
Config: tr: 0.7 - tp: 0.6
Precision = 0.6495 - Recall = 0.5463 - Fscore = 0.5934

Config: tr: 0.8 - tp: 0.4
Precision = 0.8728 - Recall = 0.7076 - Fscore = 0.7840
```

[官网精度](https://github.com/princewang1994/TextSnake.pytorch/blob/master/README.md)
```
tr=0.7 / tp=0.6时
Precision=0.652，Recall=0.549，F1-score=0.596

tr=0.8 / tp=0.4时
Precision=0.874，Recall=0.711，F1-score=0.784
```
>比较离线推理精度可知，精度下降在1个点之内，因此可视为精度达标
没有遇到精度不达标的问题，故不需要进行精度调试


## 7 性能对比

-   **[310性能数据](#71-310性能数据)**  
-   **[310P性能数据](#72-310P性能数据)**  
-   **[T4性能数据](#73-T4性能数据)** 
-   **[性能对比](#74-性能对比)**  

### 7.1 310P性能数据

batch1的性能：
 测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务

```
./benchmark.x86_64 -round=20 -om_path=TextSnake_bs1.om -device_id=0 -batch_size=1
```
执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果
```
[INFO] Dataset number: 19 finished cost 5.56ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_TextSnake_bs1_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 180.362samples/s, ave_latency: 5.57205ms
----------------------------------------------------------------
```
Interface throughputRate: 180.362 即是batch1 310P单卡吞吐率

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

batch1：
310P vs 310: 180.362fps > 1.2 * 147.2052fps
310P vs T4 : 180.362fps > 1.6 * 106.8923fps
性能在310P上的性能达到310的1.2倍,达到T4性能的1.6倍,性能达标
