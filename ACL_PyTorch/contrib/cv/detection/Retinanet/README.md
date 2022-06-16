# 基于detectron2训练的retinanet Onnx模型端到端推理指导
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
	-   [6.2 开源精度](#62-开源精度)
	-   [6.3 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)

## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[Retinanet论文](https://arxiv.org/abs/1708.02002)
论文提出了一个简单、灵活、通用的损失函数Focal loss，用于解决单阶段目标检测网络检测精度不如双阶段网络的问题。这个损失函数是针对了难易样本训练和正负样本失衡所提出的，使得单阶段网络在运行快速的情况下，获得与双阶段检测网络相当的检测精度。此外作者还提出了一个Retinanet用于检验网络的有效性，其中使用Resnet和FPN用于提取多尺度的特征。


### 1.2 代码地址

[cpu,gpu版detectron2框架retinanet代码](https://github.com/facebookresearch/detectron2)
commit_id:60fd4885d7cfd52d4267d1da9ebb6b2b9a3fc937

[离线推理版detectron2框架retinanet代码](https://github.com/facebookresearch/detectron2)
commit_id:60fd4885d7cfd52d4267d1da9ebb6b2b9a3fc937

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
pytorch == 1.8.0
torchvision == 0.9.0
onnx == 1.9.0
```

**注意：** 

>   转onnx的环境上pytorch需要安装1.8.0版本
>

### 2.2 python第三方库

```
numpy == 1.21.2
opencv-python == 4.6.0.66
```

**说明：** 
>   X86架构：opencv,pytorch,torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip install 包名 安装
>
>   Arm架构：opencv,pytorch,torchvision和onnx可以通过源码编译安装，其它可以通过pip install 包名 安装
>


## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

detectron2暂支持pytorch1.8导出pytorch框架的onnx，npu权重可以使用开源的detectron2加载，因此基于pytorch1.8与开源detectron2导出含npu权重的onnx。atc暂不支持动态shape小算子，可以使用大颗粒算子替换这些小算子规避，这些小算子可以在转onnx时的verbose打印中找到其对应的python代码，从而根据功能用大颗粒算子替换，onnx能推导出变量正确的shape与算子属性正确即可，变量实际的数值无关紧要，因此这些大算子函数的功能实现无关紧要，因包含自定义算子需要去掉对onnx模型的校验。


### 3.1 pth转onnx模型

1.获取pth权重文件  
[retinanet基于detectron2预训练的权重文件，下载路径：

github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

下载Retinanet项 Name R50 3x 38.7精度的model

md5sum：5bd44e2eaaabb0e1877c1c91f37ce513
2.下载detectron2源码并安装

```shell
git clone https://github.com/facebookresearch/detectron2
git reset 60fd4885d7cfd52d4267d1da9ebb6b2b9a3fc937 --hard
cd detectron2
python3.7 -m pip install -e .
```

 **说明：**  
> 安装所需的依赖说明请参考detectron2/INSTALL.md
>
> 重装pytorch后需要rm -rf detectron2/build/ **/*.so再重装detectron2

3.detectron2代码迁移，参见retinanet_detectron2.diff：

 **修改依据：**  
> 1.由于om不支持Non_Zero算子，而是用掩模筛选数据在输出onnx时会引入Non_Zero操作，因此不能使用掩模。所以就用直接输出index的torch.topk代替。修改参见retinanet_detectron2.diff  
> 2.由于onnx不支持torchvision的nms函数，因此使用自定义的om支持的nms算子
> 3.数据预处理步骤和后处理模块应当从网络中移除  
> 4.slice跑在aicpu有错误，所以改为dx = denorm_deltas[:, :, 0:1:].view(-1, 80) / wx，使其运行在aicore上  
> 5.atc转换时根据日志中报错的算子在转onnx时的verbose打印中找到其对应的python代码，然后找到规避方法解决，此外opt应该设为11，具体修改参见retinanet_detectron2.diff  
> 6.其它地方的修改原因参见精度调试与性能优化  


通过打补丁的方式修改detectron2：
```shell
patch -p1 < ../retinanet_detectron2.diff
cd ..
```
4.修改pytorch代码去除导出onnx时进行检查  
将/root/anconda3/lib/python3.7/site-packages/torch/onnx/utils.py文件的_check_onnx_proto(proto)改为pass

5.准备coco2017验证集，数据集获取参见本文第四章第一节  
在当前目录按结构构造数据集：datasets/coco目录下有annotations与val2017，annotations目录存放coco数据集的instances_val2017.json，val2017目录存放coco数据集的5000张验证图片。  
或者修改detectron2/detectron2/data/datasets/builtin.py为_root = os.getenv("DETECTRON2_DATASETS", "/root/datasets/")指定coco数据集所在的目录/root/datasets/。

6.运行如下命令，生成model.onnx
```shell
python3.7 detectron2/tools/deploy/export_model.py --config-file detectron2/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml
 --output ./ --export-method tracing --format onnx MODEL.WEIGHTS model_final.pkl MODEL.DEVICE cpu
```

### 3.2 onnx转om模型

1. 设置环境变量

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 使用atc将onnx模型
${chip_name}可通过npu-smi info指令查看，例：310P3
![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```shell
atc --model=model.onnx --framework=5 --output=retinanet_detectron2_npu --input_format=NCHW --input_shape="input0:1,3,1344,1344"  --log=debug --soc_version=Ascend${chip_name}
```
    --input_shape：输入数据的shape。  
    --output：输出的OM模型。  
    --log：日志级别。  
    --soc_version：处理器型号，Ascend310或Ascend710。  
    --soc_version：处理器型号。  

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用coco2017的5千张验证集进行测试，图片与标签分别存放在./datasets/coco/val2017与./datasets/coco/annotations/instances_val2017.json。

### 4.2 数据集预处理
1.预处理脚本retinanet_pth_preprocess_detectron2.py


2.执行预处理脚本，生成数据集预处理后的bin文件
```shell
python3.7 retinanet_pth_preprocess_detectron2.py --image_src_path= ./datasets/coco/val2017 --bin_file_path=val2017_bin --model_input_height=1344 --model_input_width=1344
```

### 4.3 生成预处理数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```shell
python3.7 get_info.py bin ./val2017_bin retinanet.info 1344 1344
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息


## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)

### 5.2 离线推理
1.设置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
```shell
./benchmark.x86_64 -model_type=vision -om_path=retinanet_detectron2_npu.om -device_id=0 -batch_size=1 -input_text_path=retinanet.info -input_width=1344 -input_height=1344 -useDvpp=false -output_binary=true
```
输出结果默认保存在当前目录result/dumpOutput_device0，模型有三个输出，每个输入对应的输出对应三个_x.bin文件
```
输出       shape                 数据类型    数据含义
output1    100 * 4               FP32       boxes
output2    100 * 1               Int32       labels
output3    100 * 1               FP32       scores
```

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

后处理统计map精度


调用retinanet_pth_postprocess_detectron2.py评测map精度：
```shell
python3.7 Fsaf_postprocess.py --bin_data_path=./result/dumpOutput_device0/ --test_annotation=fsaf_jpeg.info --det_results_path=./ret_npuinfer/ --net_out_num=3 --net_input_height==800 --net_input_width=1216 --ifShowDetObj --annotations_path=./datasets/coco/annotations/instances_val2017.json
```
--bin_data_path为benchmark推理结果，

--val2017_path为原始图片路径，

--test_annotation为图片信息文件，

--net_out_num为网络输出个数，

--net_input_height为网络高，

--net_input_width为网络高 
执行完后会打印出精度：

```
Loading and preparing results...
DONE (t=4.64s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=75.71s).
Accumulating evaluation results...
DONE (t=16.46s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.383
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.576
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.412
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.227
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.326
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.522
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.367
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.700
```

### 6.2 开源精度
[官网精度](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)

参考[detectron2框架的retinanet](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)，安装依赖PyTorch(GPU版本)与设置环境变量，在GPU上执行推理，测得GPU精度如下：
```shell
git clone https://github.com/facebookresearch/detectron2
python3.7 -m pip install -e detectron2
cd ./detectron2/tools
python3.7 train_net.py --eval-only --config-file ./detectron2/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml MODEL.WEIGHTS ./model_final.pkl
```
配置文件与权重文件分别为retinanet_R_50_FPN_3x.yaml与model_final.pkl，删除retinanet_R_50_FPN_3x.yaml的SOLVER和DATALOADER配置，root/datasets/coco下面放置coco2017验证集图片与标签（参考本文第三章第一节步骤五）
获得精度为

```
AP,AP50,AP75,APs,APm,APl
38.680,57.996,41.497,23.348,42.304, 50.318
```

### 6.3 精度对比

310上om推理box map精度为0.383，官方开源pth推理box map精度为0.387，精度下降在1个点之内，因此可视为精度达标，710上fp16精度0.383, 可视为精度达标

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
离线推理的Interface throughputRate即为吞吐量，对于310，需要乘以4，710只有一颗芯片，FPS为该值本身

batch1的性能：
310上Interface throughputRate: 2.2465，2.2465*4=8.98617即是batch1 310单卡吞吐率
310P上Interface throughputRate: 15.3104，15.3104即是batch1 310P单卡吞吐率
T4单卡吞吐率为8.63557

retinanet detectron2不支持多batch

 **性能优化：**  
> 查看profiling导出的op_statistic_0_1.csv算子总体耗时统计发现gather算子耗时最多，然后查看profiling导出的task_time_0_1.csv找到具体哪些gather算子耗时最多，通过导出onnx的verbose打印找到具体算子对应的代码，因gather算子计算最后一个轴会很耗时，因此通过转置后计算0轴规避，比如retinanet_detectron2.diff文件中的如下修改：
> ```
>    boxes_prof = boxes.permute(1, 0)
>    widths = boxes_prof[2, :] - boxes_prof[0, :]
> ```
>
