# Cascade-Mask-RCNN Onnx模型端到端推理指导

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
	-   [6.1 离线推理Acc精度统计](#61-离线推理Acc精度统计)
	-   [6.2 开源Acc精度](#62-开源Acc精度)
	-   [6.3 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[Cai Z, Vasconcelos N. Cascade r-cnn: Delving into high quality object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 6154-6162.](https://arxiv.org/abs/1712.00726)  

### 1.2 代码地址
[url=https://github.com/facebookresearch/detectron2.git](https://github.com/facebookresearch/detectron2.git)  
branch:master  
commit_id:468ae58cf49d09931788f378e4b3d4cc2f171c22

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC1
pytorch == 1.8.0
torchvision == 0.9.0
onnx == 1.8.0
```

### 2.2 python第三方库

```
numpy == 1.21.4
opencv-python == 4.2.0.34
decorator == 5.1.0
sympy == 1.9
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1. 准备pth权重文件  
使用训练好的pkl权重文件：model_final_e9d89b.pkl

下载路径：
https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

下载Other Settings项 Cascade R-CNN 1x 36.4精度的model文件

2. 下载detectron2源码并安装

```shell
git clone https://github.com/facebookresearch/detectron2
cd detectron2
git reset --hard aa8ea943411a2e5cd616e4d517215779a2ea2dad--hard
python3.7 -m pip install -e .
```

 **说明：**  
> 安装所需的依赖说明请参考detectron2/INSTALL.md
>

3. detectron2代码迁移
通过打补丁的方式修改detectron2：
```shell
patch -p1 < ../cascade_maskrcnn.patch 
cd ..
```

4. 准备coco2017验证集，数据集获取参见本文第四章第一节  

(a)方法一：在当前目录按结构构造数据集：datasets/coco目录下有annotations与val2017两个文件夹，annotations目录存放coco数据集的instances_val2017.json，val2017目录存放coco数据集的5000张验证图片.

(b)方法二：修改读取数据集路径。
```shell
vim detectron2/detectron2/data/datasets/builtin.py
```
修改os.getenv中的数据集路径，保存并退出。
```python
if __name__.endswith(".builtin"):     
    # Assume pre-defined datasets live in `./datasets`.     
    _root = os.getenv("DETECTRON2_DATASETS", "/opt/npu")//修改为数据集实际路径     
    register_all_coco(_root)     
    register_all_lvis(_root)     
    register_all_cityscapes(_root)     
    register_all_cityscapes_panoptic(_root)     
    register_all_pascal_voc(_root)     
    register_all_ade20k(_root)
```

6.运行如下命令，生成model.onnx
运行“detectron2/tools/deploy/export_model.py”脚本：
```shell
python3.7 detectron2/tools/deploy/export_model.py --config-file detectron2/configs/Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml --output ./output --export-method tracing --format onnx MODEL.WEIGHTS model_final_e9d89b.pkl MODEL.DEVICE cpu
```

### 3.2 onnx转om模型

1. 设置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2. 使用atc将onnx模型
${chip_name}可通过npu-smi info指令查看，例：310P3
![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

执行ATC命令：
```shell
atc --model=output/model.onnx \
--framework=5 \
--output=output/cascade_maskrcnn_bs1 \
--input_format=NCHW \
--input_shape="0:1,3,1344,1344" \
--out_nodes="Cast_1835:0;Gather_1838:0;Reshape_1829:0;Slice_1862:0" \
--log=debug \
--soc_version=Ascend${chip_name} \
```

参数说明：
--model：为ONNX模型文件。
--framework：5代表ONNX模型。
--output：输出的OM模型。
--input_format：输入数据的格式。
--input_shape：输入数据的shape。
--log：日志级别。
--soc_version：处理器型号。

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用coco2017的5千张验证集进行测试，图片与标签分别存放在./datasets/coco/val2017与./datasets/coco/annotations/instances_val2017.json。格式如下：
```
├──datasets 
   └── coco 
       ├──annotations 
           └──instances_val2017.json    //验证集标注信息        
       └── val2017                      // 验证集文件夹
```

### 4.2 数据集预处理
将原始数据（.jpg）转化为二进制文件（.bin）。以coco_2017数据集为例，通过缩放、均值方差手段归一化，输出为二进制文件。

执行“cascade_maskrcnn_preprocess.py”脚本。

```shell
python3.7 cascade_maskrcnn_preprocess.py \
--image_src_path=./datasets/coco/val2017 \
--bin_file_path=val2017_bin \
--model_input_height=1344 \
--model_input_width=1344
```
每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“val2017_bin”二进制文件夹。

### 4.3 生成数据集信息文件
使用benchmark推理需要输入图片数据集的info文件，用于获取数据集。使用get_info.py脚本，输入已经获得的图片文件，输出生成图片数据集的info文件。

1. 生成JPG图片输入info文件
```shell
python3.7 get_info.py jpg ./datasets/coco/val2017 cascade_maskrcnn_jpeg.info
```
第一个参数为生成的数据集文件格式；第二个参数为原始数据文件相对路径；第三个参数为生成的info文件名。
运行成功后，在当前目录中生成cascade_maskrcnn_jpeg.info。

2. 生成BIN文件输入info文件
```shell
python3.7 get_info.py bin ./val2017_bin cascade_maskrcnn.info 1344 1344
```
第一个参数为生成的数据集文件格式，第二个参数为预处理后的数据文件相对路径，第三个参数为生成的数据集文件名，第四个和第五个参数分别为模型输入的宽度和高度。
运行成功后，在当前目录中生成cascade_maskrcnn.info。

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
增加benchmark.{arch}可执行权限
```shell
chmod u+x benchmark.x86_64
```
执行推理
```shell
./benchmark.x86_64 -model_type=vision -om_path=output/cascade_maskrcnn_bs1.om -device_id=0 -batch_size=1 -input_text_path=cascade_maskrcnn.info -input_width=1344 -input_height=1344 -useDvpp=false -output_binary=true
```
推理后的输出默认在当前目录“result/dumpOutput_device0”下，每个输入对应的输出对应一个_x.bin文件。


## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理Acc精度统计

后处理统计Acc精度

调用cascade_maskrcnn_postprocess.py评测map精度。
```python
python3.7 cascade_maskrcnn_postprocess.py \
--bin_data_path=./result/dumpOutput_device0/ \
--test_annotation=cascade_maskrcnn_jpeg.info \
--det_results_path=./ret_npuinfer/ \
--net_out_num=4 \
--net_input_height=1344 --net_input_width=1344 \
--ifShowDetObj
```
参数说明：
--bin_data_path：为benchmark推理结果。
--test_annotation：为原始图片信息文件。
--det_results_path：为后处理输出结果。
--net_out_num：为网络输出个数。
--net_input_height、--net_input_width：为网络高宽。
--ifShowDetObj：为是否将box画在图上显示。

执行完后得到310P上的精度：

|AP  |AP50  |AP75  |APs  |APm  |APl  |
|---|---|---|---|---|---|
|36.298  |56.864  |39.031  |17.554   |38.606  |52.503  |


### 6.2 开源TopN精度
[官网精度](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)
参考[detectron2框架在线推理指南](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html)，安装依赖PyTorch(GPU版本)与设置环境变量，在GPU上执行推理，测得GPU精度如下：

```shell
git clone https://github.com/facebookresearch/detectron2

python3.7 -m pip install -e detectron2

cd ./detectron2/tools

python train_net.py --config-file ./detectron2/configs/Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml --eval-only MODEL.WEIGHTS model_final_e9d89b.pkl
```
配置文件与权重文件分别为cascade_mask_rcnn_R_50_FPN_1x.yaml与model_final_e9d89b.pkl。

root/datasets/coco下面放置coco2017验证集图片与标签（参考本文第三章第一节步骤五）

获得官网精度为：

|AP  |AP50  |AP75  |APs  |APm  |APl  |
|---|---|---|---|---|---|
|36.403  |56.945  |39.208  |17.477   |38.669  |52.491  |

### 6.3 精度对比
将得到的om离线模型推理Acc精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。

**精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
离线推理的Interface throughputRate即为吞吐量，对于310，需要乘以4，710只有一颗芯片，FPS为该值本身。

310上Interface throughputRate: ，1.18511x4=4.74044即batch1 310单卡吞吐率为4.74044。

310P上Interface throughputRate: 8.82862 ，即是batch1 310P单卡吞吐率为8.82862。

T4单卡吞吐率为4.53933。

Cascade M-RCNN不支持多batch。

**性能优化：**  

>没有遇到性能不达标的问题，故不需要进行性能优化
