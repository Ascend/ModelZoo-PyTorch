# Cascade-RCNN-Resnet101-FPN-DCN ONNX模型端到端推理指导
- [Cascade-RCNN-Resnet101-FPN-DCN ONNX模型端到端推理指导](#cascade-rcnn-resnet101-fpn-dcn-onnx模型端到端推理指导)
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
		- [5.1 获取ais_infer推理工具](#51-获取ais_infer推理工具)
		- [5.2 离线推理](#52-离线推理)
	- [6 精度对比](#6-精度对比)
		- [6.1 离线推理精度统计](#61-离线推理精度统计)
		- [6.2 开源精度](#62-开源精度)
		- [6.3 精度对比](#63-精度对比)
	- [7 性能对比](#7-性能对比)
		- [7.1 npu性能数据](#71-npu性能数据)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[Cascade R-RFD论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Deformable_ConvNets_V2_More_Deformable_Better_Results_CVPR_2019_paper.html)  
Cascade-RCNN-Resnet101-FPN-DCN是利用Deformable Conv（可变形卷积）和Deformable Pooling（可变形池化）来解决模型的泛化能力比较低、无法泛化到一般场景中、因为手工设计的不变特征和算法对于过于复杂的变换是很难的而无法设计等的问题，使用额外的偏移量来增强模块中空间采样位置，不使用额外的监督，来从目标任务中学习这个偏移量。新的模块可以很容易取代现存CNN中的counterparts，很容易使用反向传播端到端训练，形成新的可变形卷积网络。

### 1.2 代码地址
[cpu,gpu版Cascade-RCNN-Resnet101-FPN-DCN代码](https://github.com/open-mmlab/mmdetection.git)   
branch=master   
commit_id=42569a77afc3d5e67cc62c47b64087e8066494bc
## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC2
pytorch == 1.7.0
torchvision == 0.9.0
onnx == 1.7.0
onnxruntime==1.9.0
```

**注意：** 
>   转onnx的环境上pytorch需要安装1.7.0版本
>

### 2.2 python第三方库

```
numpy == 1.21.6
mmdet == 2.8.0
opencv-python == 4.5.4.60
mmpycocotools == 12.0.3
protobuf == 3.20.0
mmcv == 1.2.4
```

**说明：** 
>   X86架构：opencv,pytorch,torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：opencv,pytorch,torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.获取pth权重文件  
[Cascade-RCNN-Resnet101-FPN-DCN预训练的权重文件](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/detection/Cascade%20RCNN-Resnet101-FPN-DCN/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-3b2f0594.pth)

2.下载mmdetection源码并安装
```shell
git clone https://github.com/open-mmlab/mmdetection.git   
cd mmdetection  
git reset a21eb25535f31634cef332b09fc27d28956fb24b --hard
pip install -v -e .
cd ..
```

3.修改mmdetection源码适配Ascend NPU  
将提供的**diff补丁和pytorch_code_change**文件夹中的文件替换原文件。
```
patch -p1 < ../cascadeR101dcn_mmdetection.diff
cd ..
cp ./pytorch_code_change/deform_conv.py /root/anaconda3/envs/dcn(根据实际情况修改)/lib/python3.7/site-packages/mmcv/ops/deform_conv.py
```

4.运行如下命令，生成model.onnx  
使用mmdet框架自带的脚本导出onnx即可，这里指定shape为1216。
由于当前框架限制，仅支持batchsize=1的场景。
```shell
python mmdetection/tools/pytorch2onnx.py mmdetection/configs/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py ./cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-3b2f0594.pth --output-file=cascadeR101dcn.onnx --shape=1216 --verify --show
```

### 3.2 onnx转om模型

1.设置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
该命令中使用CANN默认安装路径(/usr/local/Ascend/ascend-toolkit)中的环境变量，使用过程中请按照实际安装路径设置环境变量。

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)，需要指定输出节点以去除无用输出，使用netron开源可视化工具查看具体的输出节点名：
${chip_name}可通过npu-smi info指令查看，例：310P3
![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)
```shell
atc --model=cascadeR101dcn.onnx  --framework=5 --output=cascadeR101dcn --input_format=NCHW --input_shape="input:1,3,1216,1216" --log=info --soc_version=Ascend${chip_name} --out_nodes="Concat_1427:0;Reshape_1429:0"
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用COCO官网的coco2017的5千张验证集进行测试，图片与标签分别存放在coco/val2017/与coco/annotations/instances_val2017.json。

### 4.2 数据集预处理
```shell
python mmdetection_coco_preprocess.py --image_folder_path ./data/coco/val2017 --bin_folder_path val2017_bin
```

### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py
2.执行生成数据集信息脚本，生成数据集信息文件
```shell
python get_info.py bin ./val2017_bin coco2017.info 1216 1216
python get_info.py jpg data/coco/val2017 coco2017_jpg.info
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息

## 5 离线推理

-   **[获取ais_infer推理工具](#51-获取ais_infer推理工具)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 获取ais_infer推理工具

https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer
本推理工具编译需要安装好CANN环境
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
git clone https://gitee.com/ascend/tools.git
cd tools/ais-bench_workload/tool/ais_infer
pip install -r requirements.txt
cd backend/
pip3.7 wheel ./
pip3 install *.whl
```

### 5.2 离线推理
1.设置环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
该命令中使用CANN默认安装路径(/usr/local/Ascend/ascend-toolkit)中的环境变量，使用过程中请按照实际安装路径设置环境变量。

2.执行离线推理
```
cd ..
# om文件和val2017_bin文件夹按实际路径填写
python3 ais_infer.py --model Cascade-RCNN-Resnet101-FPN-DCN/cascadeR101dcn.om --input Cascade-RCNN-Resnet101-FPN-DCN/val2017_bin/ --output result
```
之后会在--output指定的文件夹result生成保存推理结果的文件夹，将其重命名为infer_result

3.推理结果展示
本模型提供后处理脚本，将二进制数据转化为txt文件，同时生成画出检测框后的图片。执行脚本
```
python3 mmdetection_coco_postprocess.py --bin_data_path=result/infer_result --prob_thres=0.05 --ifShowDetObj --det_results_path=detection-results --test_annotation=coco2017_jpg.info
```

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

```
python txt_to_json.py
python coco_eval.py --ground_truth coco/annotation/instances_val2017.json
```
可以看到NPU精度：'bbox_mAP': 0.452


### 6.2 开源精度
[官网精度](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn) mAP:0.45

### 6.3 精度对比
om推理box map精度为0.452，GPU推理box map精度为0.45，精度下降在1个点之内，因此可视为精度达标

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
batch1的性能：
 测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务

统计吞吐率与其倒数时延，npu性能是一个device执行的结果
310上Interface throughputRate: ，0.673411*4=2.693644即batch1 310单卡吞吐率为2.693644。

310P上Interface throughputRate: 3.703812 ，即是batch1 310P单卡吞吐率为3.703812。

T4单卡吞吐率为3.9714058。

cascadeR101dcn不支持多batch，故只测试batch1的性能  
