# GFocalV2模型PyTorch离线推理指导

- [GFocalV2模型PyTorch离线推理指导](#GFocalV2模型PyTorch离线推理指导)
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
		- [5.1 msame工具概述](#51-msame工具概述)
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
[GfocalV2论文](https://arxiv.org/abs/1807.01544)
论文主要引入边界框不确定性的统计量来高效地指导定位质量估计，从而提升one-stage的检测器性能

### 1.2 代码地址

[GfocalV2 Pytorch实现代码](https://github.com/implus/GFocalV2)
```
branch=master 
commit_id=bfcc2b9fbbcad714cff59dacc8fb1111ce381cda
```

## 2 环境说明 

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN == 5.1.RC2
pytorch == 1.8.0
torchvision == 0.9.0
onnx == 1.9.0
```

### 2.2 python第三方库

```
mmcv-full == 1.2.4 
mmdet == 2.6.0
opencv-python == 4.5.1.48
numpy == 1.21.6
pillow == 7.2.0
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

[gfocalv2预训练的pth权重文件](https://drive.google.com/file/d/1wSE9-c7tcQwIDPC6Vm_yfOokdPfmYmy7/view?usp=sharing)

2.获取GFocalV2源码

获取GFocalV2代码
```
git clone https://github.com/implus/GFocalV2.git -b master
cd GFocalV2
git reset --hard b7b355631daaf776e097a6e137501aa27ff7e757
patch -p1 < ../GFocalV2.diff
python3.7 setup.py develop
cd ..
```
3.使用./GFocalV2/tools/pytorch2onnx.py进行onnx的转换，在目录下生成gfocal.onnx

```
python3.7 ./GFocalV2/tools/pytorch2onnx.py ./GFocalV2/configs/gfocal/gfocal_r50_fpn_1x.py ./gfocal_r50_fpn_1x.pth --output-file gfocal.onnx --input-img ./GFocalV2/demo/demo.jpg --shape 800 1216 --show
```



### 3.2 onnx转om模型

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23310P424%7C251366513%7C22892968%7C251168373)，需要指定输出节点以去除无用输出，可以使用netron开源可视化工具查看具体的输出节点名：

使用atc将onnx模型转换为om模型 ${chip_name}可通过npu-smi info指令查看

![输入图片说明](https://images.gitee.com/uploads/images/2022/0704/095450_881600a3_7629432.png "屏幕截图.png")

执行ATC命令

```shell
atc --framework=5 --model=./gfocal.onnx --output=gfocal_bs1 --input_format=NCHW --input_shape="input.1:1,3,800,1216" --log=debug --soc_version=Ascend${chip_name}
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
该模型使用COCO数据集
[coco2017](https://cocodataset.org/#download)，下载其中val2017图片及其标注文件，放入服务器/root/dataset/coco/文件夹，val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json，文件目录结构如下：
```
root
├── dataset
│   ├── coco
│   │   ├── annotations
│   │   ├── val2017
```
### 4.2 数据集预处理
1.预处理脚本gfocal_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件

```shell
python3.7 gfocal_preprocess.py --image_src_path=${datasets_path}/coco/val2017 --bin_file_path=val2017_bin --model_input_height=800 --model_input_width=1216
```

### 4.3 生成预处理数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```shell
python3.7 get_info.py jpg ${datasets_path}/coco/val2017 gfocal_jpeg.info
```
第一个参数为模型输入的类型，第二个参数为数据集路径，第三个为输出的info文件


## 5 离线推理 
-   **[msame工具概述](#51-msame工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 msame工具概述

msame工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[参考链接](https://gitee.com/ascend/tools/tree/master/msame#https://gitee.com/link?target=https%3A%2F%2Fobs-book.obs.cn-east-2.myhuaweicloud.com%2Fcjl%2Fmsame.zip)

### 5.2 离线推理
1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.将编译完成的可执行文件放到当前目录下，执行离线推理，执行时使npu-smi info查看设备状态，确保device空闲

```shell
./tools/msame/out/msame --model "./gfocal_bs1.om" --input "./val2017_bin" --output "./out/" --outfmt TXT
```

## 6 精度对比 
调用gfocal_postprocess.py：
```shell
python3.7 gfocal_postprocess.py --bin_data_path=./out/2* --test_annotation=gfocal_jpeg.info --net_out_num=3 --net_input_height=800 --net_input_width=1216
```
参数"--bin_data_path"为推理生成结果文件的路径。
执行完后会打印出精度。
```
Average Precision(AP)@[ IoU=0.50:0.95 | area=all | maxDets=100 ] =0.406
```
[官网精度](https://github.com/implus/GFocalV2)
```
Average Precision(AP)@[ IoU=0.50:0.95 | area=all | maxDets=100 ] =0.410
```

## 7 性能对比

-   **[310性能数据](#71-310性能数据)**  
-   **[310P性能数据](#72-310P性能数据)**  
-   **[T4性能数据](#73-T4性能数据)** 
-   **[性能对比](#74-性能对比)**  

### 7.1 310性能数据

batch1的性能：
 测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务

```
./tools/msame/out/msame --model "./gfocal_bs1.om" --input "./val2017_bin" --output "./out/" --outfmt TXT
```
执行数据集推理，推理完成时显示推理时间。
```
Inference average time : 223.21ms
Inference average time without first time : 223.22ms
```
由于310为四芯片，计算fps时使用4*1000/223.21=17.92fps

### 7.2 310P性能数据

batch1的性能：
 测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务

```
./tools/msame/out/msame --model "./gfocal_bs1.om" --input "./val2017_bin" --output "./out/" --outfmt TXT
```
执行数据集推理，推理完成时显示推理时间。
```
Inference average time : 28.99ms
Inference average time without first time : 28.99ms
```
计算fps时使用1000/28.99=34.49fps

### 7.3 性能对比

**评测结果：** 

| 模型      |  310P性能    | 310性能    |T4性能  |310P/310 |310P/T4| 
| :------: |  :------:  | :------:  | :------:  |:------:  |  :------:  | 
| GFocalV2 bs1  |  34.49fps |17.92fps | 9.4fps |1.92| 3.67|

```
# 710性能是否超过基准： 是
310P vs 310: bs1:710=(34.49/17.92) 1.92倍基准
310P vs T4: bs1:710=(34.49/9.4) 3.67倍基准
性能在310P上的性能超过310的1.2倍,超过T4性能的1.6倍,性能达标
备注：离线模型不支持多batch。
```