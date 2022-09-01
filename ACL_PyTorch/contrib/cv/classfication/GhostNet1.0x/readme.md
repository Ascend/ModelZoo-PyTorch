# GhostNet1.0x模型PyTorch离线推理指导

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
	-   [5.1 获取ais_infer推理工具](#51-获取ais_infer推理工具)
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
[GhostNet论文](https://arxiv.org/abs/1911.11907)  

### 1.2 代码地址
[GhostNet代码](https://github.com/huawei-noah/CV-Backbones/tree/master/ghostnet_pytorch)
branch:master
commit_id:5a06c87a8c659feb2d18d3d4179f344b9defaceb

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC1
pytorch >= 1.8.0
torchvision >= 0.9.0
onnx >= 1.7.0
```

### 2.2 python第三方库

```
numpy == 1.18.5
Pillow == 8.2.0
opencv-python == 4.5.1.48
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
[GhostNet预训练pth权重文件](https://github.com/huawei-noah/CV-Backbones/raw/master/ghostnet_pytorch/models/state_dict_73.98.pth)   
文件md5sum:   F7241350B4486BF00ACCBF9C3A192331

```
wget http://github.com/huawei-noah/CV-Backbones/raw/master/ghostnet_pytorch/models/state_dict_73.98.pth
```

2.GhostNet模型代码从如下代码仓中下载
https://github.com/huawei-noah/CV-Backbones/tree/master/ghostnet_pytorch

3.编写pth2onnx脚本ghostnet_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 ghostnet_pth2onnx.py state_dict_73.98.pth ghostnet.onnx
```

 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明

### 3.2 onnx转om模型

1.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)
```
atc --framework=5 --model=./ghostnet.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=ghostnet_bs1 --log=debug --soc_version=Ascend${chip_name}
```
运行成功后生成“ghostnet_bs1.om”模型文件。

参数说明： 

--model：为ONNX模型文件

--framework：5代表ONNX模型

--output：输出的OM模型

--input_format：输入数据的格式

--input_shape：输入数据的shape

--log：日志级别

--soc_version：处理器型号

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/root/datasets/imagenet/val与/root/datasets/imagenet/val_label.txt。

### 4.2 数据集预处理
1.预处理脚本imagenet_torch_preprocess.py

2.如验证的数据集在目录imageNet下
```
├── imageNet    
       └── val       // 验证集文件夹
├── val_label.txt    //验证集标注信息      
```
执行预处理脚本，生成数据集预处理后的bin文件，将原始数据（.jpg）转化为二进制文件（.bin）存放在prep_dataset文件目录下
```
python3.7 imagenet_torch_preprocess.py ghostnet /root/datasets/imageNet/val ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 gen_dataset_info.py bin ./prep_dataset ./ghostnet_prep_bin.info 224 224
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
## 5 离线推理

-    **[获取ais_infer推理工具](#51-获取ais_infer推理工具)** 

-   **[离线推理](#52-离线推理)**  

### 5.1 获取ais_infer推理工具

https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer

将工具编译后的压缩包放置在当前目录；解压工具包，安装工具压缩包中的whl文件； pip3 install aclruntime-0.01-cp37-cp37m-linux_xxx.whl
### 5.2 离线推理
昇腾芯片上执行，执行时使npu-smi info查看设备状态，确保device空闲

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
```
python ais_infer.py --model ./ghostnet_bs1.om --input ./prep_dataset/ --output ./ --outfmt NPY --batchsize 1
```
参数说明： 

--model：为OM模型文件

--input：为数据路径

--output：输出推理结果

--outfmt：输出结果的格式

--input_shape：输入数据的shape

--batchsize：模型接受的bs大小

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

后处理统计TopN精度

调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得精度结果数据，显示在控制台。
```
python3.7 imagenet_acc_eval.py ./lcmout/2022_xx_xx-xx_xx_xx/sumary.json /home/HwHiAiUser/dataset/imageNet/val_label.txt
```
参数说明：

第一项参数为推理结果中的sumary.json文件，第二项为gt标签文件

### 6.2 开源TopN精度
[ghostnet代码仓公开模型精度](https://github.com/huawei-noah/CV-Backbones/tree/master/ghostnet_pytorch)
```
Model           Acc@1     Acc@5
ghostnet    	73.98     91.46
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据  
1.ais_infer工具在整个数据集上推理获得性能数据  
比如batch_16的性能结果，ais_infer工具在整个数据集上推理后生成lcmout/2022_xx_xx-xx_xx_xx/sumary.json：   
```
"NPU_compute_time": {"min": 3.8949999809265137, 
		"max": 81.9729995727539, 
		"mean": 4.652794558563232, 
		"median": 3.9760000705718994, 
"percentile(99%)": 9.564919853210439}, 
		"H2D_latency": {"min": 1.6279220581054688, 
		"max": 43.206214904785156, 
		"mean": 4.154865646362305, 
		"median": 1.985788345336914, 
		"percentile(99%)": 18.392877578735302}, 
"D2H_latency": {"min": 0.045299530029296875, 
		"max": 17.41957664489746, 
		"mean": 0.7445654296875, 
		"median": 0.10347366333007812, 
		"percentile(99%)": 5.664710998535149}, 
"throughput": 3438.7935677393734}
```
throughput: 3438.7935677393734既是在bs16下的310P的单卡吞吐率  

2.各个bs下的性能对比：

| Batch Size | 310 | 310P | t4 | 310P/310| 310P/t4|
|:------|:------:|:------:|:------:|:------:|:------:|
| 1 | 	1348.024 | 1502.4291 | 219.2172| 1.1145 | 6.8536|
| 4 | 2233.9991 | 2317.6152 | 701.0072 | 1.0374| 3.3061|
| 8 | 2463.9302 | 3739.9555| 1032.52| 1.5179 | 3.6222|
| 16 | 2624.8900 | 3438.7936| 924.992| 1.3101| 3.7176|
| 32 | 2689.0490 | 3020.9916| 447.872| 1.1234| 6.7452|
 **性能优化：**  
没有遇到性能不达标的问题，故不需要进行性能优化