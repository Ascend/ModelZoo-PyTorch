# ErfNet模型PyTorch离线推理指导

- [ErfNet模型PyTorch离线推理指导](#vnet-onnx模型端到端推理指导)

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
    - [5.1 ais_bench工具概述](#51-ais_bench工具概述)
    - [5.2 离线推理](#52-离线推理)
  - [6 精度对比](#6-精度对比)
    - [6.1 离线推理精度](#61-离线推理精度)
    - [6.2 开源精度](#62-开源精度)
    - [6.3 精度对比](#63-精度对比)
  - [7 性能对比](#7-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址

[ErfNet论文](https://ieeexplore.ieee.org/abstract/document/8063438)  

### 1.2 代码地址

[ErfNet代码](https://github.com/Eromera/erfnet_pytorch)  
branch:master  
commit_id=d4a46faf9e465286c89ebd9c44bc929b2d213fb3 
备注：commit_id是指基于该次提交时的模型代码做推理，通常选择稳定版本的最后一次提交，或代码仓最新的一次提交  

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架

```
CANN 5.1.RC1
pytorch >= 1.5.0
torchvision >= 0.6.0
onnx >= 1.7.0
```

### 2.2 python第三方库

```
numpy == 1.20.2
Pillow == 7.2.0
opencv-python == 4.5.2.52
```

**说明：** 

>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.获取，修改与安装开源模型代码 

```
git clone https://github.com/Eromera/erfnet_pytorch   
cd erfnet_pytorch  
git reset d4a46faf9e465286c89ebd9c44bc929b2d213fb3 --hard
cd ..  
```

2.获取权重文件放到当前目录

[erfnet_pretrained.pth](https://github.com/Eromera/erfnet_pytorch/blob/master/trained_models/erfnet_pretrained.pth)   

3.执行ErfNet_pth2onnx.py脚本，生成onnx模型文件，由于使用原始的onnx模型转出om后，精度有损失，故添加了modify_bn_weights.py来修改转出onnx模型bn层的权重。

```
python ErfNet_pth2onnx.py erfnet_pretrained.pth ErfNet_origin.onnx
python modify_bn_weights.py ErfNet_origin.onnx ErfNet.onnx
```

### 3.2 onnx转om模型

1.设置环境变量

```
source /usr/local/Ascend/ascend-set_env.sh
```

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN 5.0.1 开发辅助工具指南 (推理) 01]

```
atc --framework=5 --model=ErfNet.onnx --output=ErfNet_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,512,1024" --log=debug --soc_version=Ascend{chip_name} --output_type=FP16
```

“{chip_name}”：npu处理器型号。（请使用npu-smi info指令查询）

![1660301955987](C:\Users\86188\AppData\Roaming\Typora\typora-user-images\1660301955987.png)

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  


### 4.1 数据集获取

[获取cityscapes](https://www.cityscapes-dataset.com/)

- Download the Cityscapes dataset from https://www.cityscapes-dataset.com/

  - Download the "leftImg8bit" for the RGB images and the "gtFine" for the labels.
  - Please note that for training you should use the "_labelTrainIds" and not the "_labelIds", you can download the [cityscapes scripts](https://github.com/mcordts/cityscapesScripts) and use the [conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) to generate trainIds from labelIds  

### 4.2 数据集预处理

执行“ErfNet_preprocess.py”脚本将原始数据集转换为模型输入的数据。

```
python ErfNet_preprocess.py ${datasets_path}/cityscapes/leftImg8bit/val ./prep_dataset ${datasets_path}/cityscapes/gtFine/val ./gt_label
```

{datasets_path}/cityscapes/leftImg8bit/val：数据集路径。（请用 数据集准确路径替换{datasets_path}）

./prep_dataset：输出文件路径。

${datasets_path}/cityscapes/gtFine/val：数据集路径。（请用 数据集准确路径替换{datasets_path}）

./gt_label：输出文件路径。

## 5 离线推理

-   **[ais_bench工具概述](#51-ais_bench工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 ais_bench工具概述

 AisBench推理工具，该工具包含前端和后端两部分。 后端基于c+开发，实现通用推理功能； 前端基于python开发，实现用户界面功能 

### 5.2 离线推理

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.执行离线推理

```
python3 -m ais_bench --model ${user_path}/ErfNet/ErfNet_bs1.om --input=${user_path}/ErfNet/prep_dataset/ --outfmt BIN --output ${user_path}/output/ --batchsize 1
```

{user_path}：请用用户个人文件准确路径替换。

输出结果保存在 ${user_path}/output/下面，每个输入对应一个_X.bin文件的输出。

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度

后处理统计精度

执行后处理脚本进行精度验证。

```
python ErfNet_postprocess.py ${user_path}/output/2022_07_15-14_16_46/sumary.json ${user_path}/ErfNet/gt_label/
```

“${user_path}/output/2022_07_15-14_16_46/sumary.json”：ais_infer推理结果汇总数据保存路径。

${user_path}/ErfNet/gt_label/：合并后的验证集路径。

310精度测试结果

```
iou is  tensor(0.7220, dtype=torch.float64)
```

310p精度测试结果

```
iou is  tensor(0.7220, dtype=torch.float64)
```

经过对batchsize1/4/8/16/32/64的om测试，精度数据均如上。

### 6.2 开源精度

[官网pth精度](https://github.com/Eromera/erfnet_pytorch)

```
iou:72.20
```

### 6.3 精度对比

| batch | 310   | 310P  |
| ----- | ----- | ----- |
| 1     | 72.20 | 72.20 |
| 4     | 72.20 | 72.20 |
| 8     | 72.20 | 72.20 |
| 16    | 72.20 | 72.20 |
| 32    | 72.20 | 72.20 |

将得到的om离线模型推理IoU精度与该模型github代码仓上公布的精度对比，310与710上的精度下降在1%范围之内，故精度达标。  
 **精度调试：**  

>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

310 310P T4性能对比如下

| batchsize | T4       | 310          | 310P    | 310P/310 | 310P/T4 | 310P-AOE | 310P-AOE/310 | 310P-AOE/T4 |
| --------- | -------- | ------------ | ------- | -------- | ------- | -------- | ------------ | ----------- |
| 1         | 215.8135 | 220.2016     | 292.137 | 1.327    | 1.35    | 380.0822 | 1.73         | 1.76        |
| 4         | 226.745  | 176.2568     | 192.398 | 1.09     | 0.849   | 379.2337 | 2.15         | 1.67        |
| 8         | 237.8234 | 176.6912     | 210.707 | 1.1925   | 0.886   | 381.9549 | 2.16         | 1.6         |
| 16        | 250.2417 | 175.8732     | 211.125 | 1.2      | 0.84    | 379.9660 | 2.16         | 1.51        |
| 32        | 222.312  | 181.5056     | 215.234 | 1.186    | 0.9682  | 380.1598 | 2.09         | 1.71        |
| 64        | 226.2459 | 内存分配失败 | 216.937 | /        | 0.96    | 226.5988 | /            | 1.001       |
| 最优batch | 250.2417 | 220.2016     | 292.137 | 1.33     | 1.17    | 381.9545 | 1.73         | 1.53        |

经过对比，AOE调优后的性能结果已经达到交付要求。

备注：  

1.由于使用原始的onnx模型转出om后，精度有损失，故添加了modify_bn_weights.py来修改转出onnx模型bn层的权重。
2.由于tensorRT不支持部分算子，故gpu性能数据使用在线推理的数据。