# RFCN Onnx模型端到端推理指导
-   [1 模型概述](#1-模型概述)
	-   [1.1 论文地址](#11-论文地址)
	-   [1.2 代码地址](#12-代码地址)
-   [2 环境说明](#2-环境说明)
	-   [2.1 深度学习框架](#21-深度学习框架)
	-   [2.2 环境搭建](#22-环境搭建)
-   [3 模型转换](#3-模型转换)
	-   [3.1 pth转onnx模型](#31-pth转onnx模型)
	-   [3.2 onnx转om模型](#32-onnx转om模型)
-   [4 数据集预处理](#4-数据集预处理)
-   [5 离线推理](#5-离线推理)
	-   [5.1 benchmark工具概述](#51-benchmark工具概述)
	-   [5.2 设置可执行权限](#52-设置可执行权限)
	-   [5.2 离线推理](#53-离线推理)
-   [6 精度对比](#6-精度对比)
	-   [6.1 离线推理精度统计](#61-离线推理精度统计)
	-   [6.2 T4精度](#62-T4精度)
	-   [6.3 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)
	-   [7.2 T4性能数据](#72-T4性能数据)
	-   [7.3 性能对比](#73-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[RFCN论文](https://arxiv.org/abs/1605.06409)  
RFCN基于faster rcnn的基础上对roi pooling这部分进行了改进，与之前的基于区域的检测器相比，此模型的基于区域的检测器是完全卷积的，几乎所有计算都在整个图像上共享，为了实现这一目标，提出了位置敏感得分图来解决图像分类中的平移不变性和目标检测中的平移可变性之间的困境。

### 1.2 代码地址
[RFCN代码](https://github.com/RebornL/RFCN-pytorch.1.0)  

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[环境搭建](#22-环境搭建)**  

### 2.1 深度学习框架
```
pytorch == 1.7.0
torchvision == 0.8.0
onnx >= 1.10.0
```

### 2.2 环境搭建

1.下载RFCN模型代码
```
git clone https://github.com/RebornL/RFCN-pytorch.1.0

```

2.补丁

```
patch -re -p0 < RFCN.patch
```

3.下载pth权重文件  

RFCN预训练pth权重文件：[faster_rcnn_2_12_5010.pth]

链接：https://pan.baidu.com/s/1HcGAXmDTLbVm_m5M1T3A3A 
提取码：er9jg

放到RFCN文件夹中

4.安装第三方库
```
cd RFCN-pytorch.1.0
pip install -r requirements.txt
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
```
5.编译
```
cd lib
python setup.py build develop
cd ..

```

6.在RFCN-pytorch.1.0目录下创建data文件夹
```
cd RFCN-pytorch.1.0
mkdir data
```

7.coco
```
cd data
git clone https://github.com/pdollar/coco.git 
cd coco/PythonAPI
make
cd ../../..
```

8.安装MagicONNX
```
git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
cd MagicONNX
pip install .
```

9.数据集获取
用户可自行准备好数据集，选用VOCtest_06-Nov-2007，解压后放入data文件夹下
```
RFCN-pytorch.1.0/data/VOCdevkit2007/VOC2007/
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型


1.执行pth2onnx脚本，生成onnx模型文件
```
python rfcn_pth2onnx.py  --input ./faster_rcnn_2_12_5010.pth --output rfcn_1.onnx

```

2.对生成的onnx进行脚本修改(第一个参数是输入的onnx文件，第二个参数是输出的之后的onnx文件)
```
python rfcn_adapt.py rfcn_1.onnx rfcn_final.onnx
```

### 3.2 onnx转om模型

1.设置环境变量
```
打开env.sh
复制内容后粘贴到命令行
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)
```
atc --framework=5 --model=./rfcn_final.onnx --output=rfcn --input_format=NCHW --input_shape="im_data:1,3,1344,1344" --log=debug --soc_version=Ascend310

```

## 4 数据集预处理

-   **[数据集预处理](#42-数据集预处理)**  


执行预处理脚本，生成数据集预处理后的bin文件(第一个参数是输入的图片路径，第二个参数是输出之后的bin文件存放路径,第三个参数是输出之后的info文件存放路径)
```
python rfcn_preprocess.py --file_path ./RFCN-pytorch.1.0/data/VOCdevkit2007/VOC2007/JPEGImages/ --bin_path ./bin --info_name demo.info

```

## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  
-   **[设置可执行权限](#52-设置可执行权限)**  
-   **[离线推理](#53-离线推理)**  

### 5.1 benchmark工具概述
benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考(https://support.huawei.com/enterprise/zh/doc/EDOC1100191895)

### 5.2 设置可执行权限
```
chmod +x benchmark.${arch}
```
### 5.3 离线推理

```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=rfcn.om -input_text_path=./demo.info -input_width=1344 -input_height=1344 -output_binary=True -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_device0,模型有三个输出，每个输入对应的输出对应三个bin文件；
性能数据默认保存在result/perf_vision_batchsize_1_device_0.txt

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

后处理统计map精度

```
python rfcn_postprocess.py --input ./result/dumpOutput_device0/ --output out
```

查看输出结果：
```
Mean AP = 0.6993

```

### 6.2 GPU精度
```
python test.py --cuda
```
```
Model        mAP     
RFCN       0.6986	   
```
### 6.3 精度对比
将得到的om离线模型推理精度比直接跑源码仓的推理脚本的精度要高，故精度达标。  


## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[T4性能数据](#72-T4性能数据)**  
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据

1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能  
```
[inference] Interface throughputRate: 1.23284
```
1.23284 * 4 = 4.92 即是batch 1 310单卡的性能数据

### 7.2 T4性能数据

跑的是源码推理不涉及batchsize,在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务

```
python test.py --cuda
```


推理总时长：1370s 共4952张图片 吞吐率为：4952/1370=3.61


### 7.3 性能对比

npu吞吐率为T4的1.36倍，符合要求
