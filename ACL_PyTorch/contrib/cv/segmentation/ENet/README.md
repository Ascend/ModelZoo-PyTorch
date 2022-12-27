# ENet Onnx模型端到端推理指导

- [ENet Onnx模型端到端推理指导](#enet-onnx模型端到端推理指导)
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
        - [5.1 AisBench工具概述](#51-aisbench工具概述)
        - [5.2 离线推理](#52-离线推理)
    - [6 精度对比](#6-精度对比)
        - [6.1 离线推理MIoU精度](#61-离线推理miou精度)
        - [6.2 精度对比](#62-精度对比)
    - [7 性能对比](#7-性能对比)
        - [7.1 npu性能数据](#71-npu性能数据)
        - [7.2 T4性能数据](#72-t4性能数据)
        - [7.3 性能对比表格](#73-性能对比表格)

## 1 模型概述

- **[论文地址](#11-论文地址)**  

- **[代码地址](#12-代码地址)**  

### 1.1 论文地址

[ENet论文](https://arxiv.org/pdf/1606.02147.pdf)  

### 1.2 代码地址

[ENet代码](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)  
branch:master  
commit_id: **5843f75215dadc5d734155a238b425a753a665d9**  
备注：commit_id是指基于该次提交时的模型代码做推理，通常选择稳定版本的最后一次提交，或代码仓最新的一次提交  

上述开源代码仓库没有给出训练好的模型权重文件，因此使用910训练好的pth权重文件来做端到端推理，该权重文件的精度是**54.627%**。

## 2 环境说明

- **[深度学习框架](#21-深度学习框架)**  

- **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架

```
CANN 5.1.RC2
pytorch >= 1.5.0
torchvision >= 0.6.0
onnx >= 1.7.0
```
本实验环境中Torch版本为1.5.0

### 2.2 python第三方库

```
numpy == 1.20.3
Pillow == 8.4.0
opencv-python == 4.5.2.54
albumentations == 0.4.5
densetorch == 0.0.2
```

**说明：**

> X86架构：pytorch和torchvision可以通过官方下载whl包安装，其他可以通过pip3.7 install 包名 安装
>
> Arm架构：pytorch，torchvision和opencv可以通过github下载源码编译安装，其他可以通过pip3.7 install 包名 安装
>
> 以上为多数网络需要安装的软件与推荐的版本，根据实际情况安装。如果python脚本运行过程中import 模块失败，安装相应模块即可，如果报错是缺少动态库，网上搜索报错信息找到相应安装包，执行apt-get install 包名安装即可

## 3 模型转换

- **[pth转onnx模型](#31-pth转onnx模型)**  

- **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.编写pth2onnx脚本RefineNet_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

2.执行pth2onnx脚本，生成onnx模型文件

```bash
python3.7 ENet_pth2onnx.py --input-file ./enet_citys.pth --output-file ./enet_citys.onnx
```

### 3.2 onnx转om模型

1.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.1.RC2 开发辅助工具指南 (推理) 01  
`${chip_name}`可通过 `npu-smi info` 指令查看，例: 310P3
![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```BASH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=./enet_citys.onnx --output=./enet_citys_bs1 --input_format=NCHW --input_shape="image:1,3,480,480" --log=info --soc_version=Ascend${chip_name}
```

## 4 数据集预处理

- **[数据集获取](#41-数据集获取)**  

- **[数据集预处理](#42-数据集预处理)**  

- **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取

该模型使用Cityscapes数据集作为训练集，其下的val中的500张图片作为验证集。推理部分只需要用到这500张验证图片，验证集输入图片存放在`citys/leftImg8bit/val`，验证集target存放在`citys/gtFine/val`。

下载Cityscapes数据集后，把数据集解压放在数据集共享文件夹`/opt/npu`下。

### 4.2 数据集预处理

1.参考开源代码仓库对验证集所做的预处理，编写预处理脚本。

2.执行预处理脚本，生成数据集预处理后的bin文件  
`$datasets_path` 为Cityscapes数据集的路径。

```bash
python3.7 ENet_preprocess.py --src-path=$datasets_path --save_path ./prep_dataset
```

## 5 离线推理

- **[AisBench工具概述](#51-AisBench工具概述)**  

- **[离线推理](#52-离线推理)**  

### 5.1 AisBench工具概述

AisBench推理工具，该工具包含前端和后端两部分。 后端基于c++开发，实现通用推理功能； 前端基于python开发，实现用户界面功能。  
工具链接: <https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer/>

### 5.2 离线推理

1.执行离线推理

```
python -m ais_bench --model ./enet_bs16.om --input ./prep_dataset/ --output ./ais_results --outfmt BIN --batchsize=16
```

--model：模型地址  
--input：预处理完的数据集文件夹  
--output：推理结果保存地址  
--outfmt：推理结果保存格式  
--batchsize：模型batch size 默认为1 。当前推理模块根据模型输入和文件输出自动进行组batch。参数传递的batchszie有且只用于结果吞吐率计算。请务必注意需要传入该值，以获取计算正确的吞吐率。  
输出结果默认保存在当前目录ais_results/X(X为执行推理的时间)，每个输入对应一个_X.bin文件的输出。  

## 6 精度对比

- **[离线推理MIoU精度](#61-离线推理IoU精度)**
- **[精度对比](#62-精度对比)**  

### 6.1 离线推理MIoU精度

后处理统计MIoU精度

调用ENet_postprocess.py脚本推理结果与语义分割真值进行比对，可以获得IoU精度数据。

```bash
python3.7 ENet_postprocess.py --src-path=$datasets_path  --result-dir ./ais_results/2022_07_11-15_53_11/sumary.json | tee eval_log.txt
```

第一个为真值所在目录，第二个为AisBench输出目录中summary.json的路径。  
`| tee eval_log.txt`  为将输出结果保存至 “eval_log.txt”的文件中。  
查看输出结果：

```
BS 1
miou: 54.115%
BS 16
miou: 54.115%
```

经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上。

### 6.2 精度对比

ENet论文给出的精度是58.3%，但它没有训练代码，也没有给出训练好的模型权重。因此只能与910训练好的模型权重进行精度对比（0.54627）。

将得到的om离线模型推理miou精度与910训练好的`.pth权重的miou进行对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  

>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

- **[npu性能数据](#71-npu性能数据)**  
 **[T4性能数据](#72-T4性能数据)**  
 **[性能对比表格](#73-性能对比表格)**  

### 7.1 npu性能数据

AisBench工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用AisBench纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，AisBench纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认AisBench工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用AisBench工具在整个数据集上推理得到bs1与bs16的性能数据为准。  

### 7.2 T4性能数据

在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2  

### 7.3 性能对比表格

|         | 310          | 310P3        | 310P3(aoe)   | T4           | 310P3/310    | 310P3(aoe)/310 | 310P3(aoe)/T4 |
|---------|--------------|--------------|--------------|--------------|--------------|----------------|---------------|
| bs1     | 825.2858383  | 731.749437   | 1041.247997  | 88.80967977  | 0.886661812  | 1.261681649    | 11.72448769   |
| bs4     | 797.8526878  | 947.2603366  | 1107.245594  | 76.70662909  | 1.187262199  | 1.387781994    | 14.43480971   |
| bs8     | 706.8480246  | 767.3814194  | 1050.013124  | 76.49302703  | 1.085638486  | 1.485486396    | 13.72691296   |
| bs16    | 701.0886768  | 642.8421665  | 1063.887262  | 77.02512213  | 0.91691991   | 1.517478883    | 13.81221129   |
| bs32    | -            | 626.778119   | 1068.416192  | 76.6339237   | -            | -              | 13.94181768   |
|         |              |              |              |              |              |                |               |
| 最优Batch | 825.2858383  | 947.2603366  | 1107.245594  | 88.80967977  | 1.147796669  | 1.341651029    | 12.46762286   |


>310P单个device的吞吐率比310单卡的吞吐率大，故310P性能高于310性能，性能达标。  
>对于batch1与batch16，310P性能均高于310性能1.2倍，该模型放在Benchmark/cv/segmentation目录下。  

 **性能优化：**  
>以上在310P上的结果为AOE优化后的性能。  
使用AOE进行性能优化

```
# 以batch size 32 为例
# 保存知识库
export TUNE_BANK_PATH=/home/HwHiAiUser/custom_tune_bank
# 执行AOE优化
aoe --framework 5 --model enet_citys.onnx --job_type 1 --output ./enet_bs32_aoe_job1_1 --input_shape="image:32,3,480,480"  --log error
```
