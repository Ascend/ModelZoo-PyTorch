# 基于开源mmsegmentation预训练的fcn-8s Onnx模型端到端推理指导
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
[fcn-8s论文](https://arxiv.org/abs/1411.4038)  
论文提出 Fully Convolutional Networks（FCN）方法用于图像语义分割，将图像级别的分类扩展到像素级别的分类，获得 CVPR2015 的 best paper。


### 1.2 代码地址
[mmsegmentation框架fcn-8s代码](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/fcn)   
branch:master commit_id:e6a8791ab0a03c60c0a9abb8456cd4d804342e92

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

### 2.2 python第三方库
```
numpy == 1.20.1
opencv-python == 4.5.2.52
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
[fcn-8s基于mmsegmentation预训练的npu权重文件](https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r50-d8_512x512_20k_voc12aug/fcn_r50-d8_512x512_20k_voc12aug_20200617_010715-52dc5306.pth)  
文件md5sum: 0b42f76eb2e3779a5f802acb5ded5eed
  
2.mmsegmentation源码安装
```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip3.7 install -e .
cd ..
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation  
如果修改了模型代码，交付了{model_name}.diff  
patch -p1 < ../{model_name}.diff
如果模型代码需要安装，则安装模型代码(如果没有安装脚本，pth2onnx等脚本需要引用模型代码的类或函数，可通过sys.path.append(r"./pytorch-nested-unet")添加搜索路径的方式)
pip3.7 install -e .  # or "python3.7 setup.py develop"
cd ..
```

 **说明：**  
> 安装所需的依赖说明请参考mmsegmentation/docs/get_started.md


3.使用tools里的pytorch2onnx.py文件，运行如下命令，生成对应的onnx模型：
```shell
python3.7 mmsegmentation/tools/pytorch2onnx.py mmsegmentation/configs/fcn/fcn_r50-d8_512x512_20k_voc12aug.py --checkpoint fcn_r50-d8_512x512_20k_voc12aug_20200617_010715-52dc5306.pth --output-file fcn_r50-d8_512x512_20k_voc12aug.onnx --shape 500 500 --show 
```
 **模型转换要点：**  
> 虽然导出的onnx可以转换为多batch的om离线推理，但是在线推理与onnx目前还不支持多batch推理

### 3.2 onnx转om模型

1.设置环境变量
```shell
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN 5.0.1 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)，如果存在多余输出节点,需要指定输出节点以去除无用输出，节点序号可能会因网络结构不同而不同，使用netron开源可视化工具查看具体的输出节点名：
```shell
atc --framework=5 --model=fcn_r50-d8_512x512_20k_voc12aug.onnx  --output=fcn_r50-d8_512x512_20k_voc12aug_bs1 --input_format=NCHW --input_shape="input:1,3,500,500" --log=debug --soc_version=Ascend310
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[VOC2012官网](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)的VOC2012的1449张验证集进行测试，图片与对应ground truth分别存放在/opt/npu/VOCdevkit/VOC2012/JPEGImages/与/opt/npu/VOCdevkit/VOC2012/SegmentationClass/。

### 4.2 数据集预处理
1.预处理脚本mmsegmentation_voc2012_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```shell
python3.7 mmsegmentation_voc2012_preprocess.py --image_folder_path=/opt/npu/VOCdevkit/VOC2012/JPEGImages/ --split=/opt/npu/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt --bin_folder_path=./voc12_bin/
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```shell
python3.7 get_info.py bin  ./voc12_bin voc12.info 500 500
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN 5.0.1 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)

### 5.2 离线推理

1.设置环境变量
```shell
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/
```
2.执行离线推理
```shell
./benchmark.${arch} -model_type=vision -om_path=fcn_r50-d8_512x512_20k_voc12aug_bs1.om -device_id=0 -batch_size=1 -input_text_path=voc12.info -input_width=500 -input_height=500 -useDvpp=false -output_binary=true
```
 **注意：**  
> onnx的输出是int64，但是om的输出是int32

输出结果默认保存在当前目录result/dumpOutput_device0，模型有一个输出，每个输入对应的输出对应_1.bin文件
```
输出       shape                 数据类型    数据含义
output1  1 * 1 * 500 * 500        int32     8位图像
```

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

1.调用mmsegmentation_voc2012_postprocess.py评测bs1的mIoU精度：
```shell
python3.7 get_info.py jpg /opt/npu/VOCdevkit/VOC2012/JPEGImages/ voc12_jpg.info

python3.7 mmsegmentation_voc2012_postprocess.py --bin_data_path=./result/dumpOutput_device0 --test_annotation=./voc12_jpg.info --img_dir=/opt/npu/VOCdevkit/VOC2012/JPEGImages --ann_dir=/opt/npu/VOCdevkit/VOC2012/SegmentationClass --split=/opt/npu/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt --net_input_width=500 --net_input_height=500
```
第一个参数为benchmark推理结果，第二个为原始图片信息文件，第三个为原始图片位置，第四个为验证图片位置，第五个图片的split，第六七个为网宽高  
执行完后会打印出精度：
```
per class results:

+-------------+-------+-------+
| Class       | IoU   | Acc   |
+-------------+-------+-------+
| background  | 92.84 | 97.27 |
| aeroplane   | 81.0  | 90.2  |
| bicycle     | 37.6  | 84.07 |
| bird        | 80.3  | 87.49 |
| boat        | 64.63 | 77.42 |
| bottle      | 61.32 | 69.76 |
| bus         | 87.31 | 91.7  |
| car         | 79.48 | 89.74 |
| cat         | 85.69 | 92.6  |
| chair       | 30.69 | 44.66 |
| cow         | 73.21 | 82.52 |
| diningtable | 43.5  | 48.95 |
| dog         | 78.83 | 87.76 |
| horse       | 74.5  | 82.18 |
| motorbike   | 75.7  | 82.97 |
| person      | 83.24 | 89.45 |
| pottedplant | 53.23 | 64.87 |
| sheep       | 74.29 | 80.85 |
| sofa        | 45.59 | 55.79 |
| train       | 77.98 | 82.49 |
| tvmonitor   | 68.21 | 74.91 |
+-------------+-------+-------+
Summary:

+--------+-------+-------+-------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+-------+
| global | 69.01 | 78.94 | 93.04 |
+--------+-------+-------+-------+

```

2.调用mmsegmentation_voc2012_postprocess.py评测bs16的mIoU精度：
```shell
python3.7 get_info.py jpg /opt/npu/VOCdevkit/VOC2012/JPEGImages/ voc12_jpg.info

python3.7 mmsegmentation_voc2012_postprocess.py --bin_data_path=./result/dumpOutput_device1 --test_annotation=./voc12_jpg.info --img_dir=/opt/npu/VOCdevkit/VOC2012/JPEGImages --ann_dir=/opt/npu/VOCdevkit/VOC2012/SegmentationClass --split=/opt/npu/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt --net_input_width=500 --net_input_height=500
```
第一个参数为benchmark推理结果，第二个为原始图片信息文件，第三个为原始图片位置，第四个为验证图片位置，第五个图片的split，第六七个为网宽高  
执行完后会打印出精度：
```
per class results:

+-------------+-------+-------+
| Class       | IoU   | Acc   |
+-------------+-------+-------+
| background  | 92.84 | 97.27 |
| aeroplane   | 81.0  | 90.2  |
| bicycle     | 37.6  | 84.07 |
| bird        | 80.3  | 87.49 |
| boat        | 64.63 | 77.42 |
| bottle      | 61.32 | 69.76 |
| bus         | 87.31 | 91.7  |
| car         | 79.48 | 89.74 |
| cat         | 85.69 | 92.6  |
| chair       | 30.69 | 44.66 |
| cow         | 73.21 | 82.52 |
| diningtable | 43.5  | 48.95 |
| dog         | 78.83 | 87.76 |
| horse       | 74.5  | 82.18 |
| motorbike   | 75.7  | 82.97 |
| person      | 83.24 | 89.45 |
| pottedplant | 53.23 | 64.87 |
| sheep       | 74.29 | 80.85 |
| sofa        | 45.59 | 55.79 |
| train       | 77.98 | 82.49 |
| tvmonitor   | 68.21 | 74.91 |
+-------------+-------+-------+
Summary:

+--------+-------+-------+-------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+-------+
| global | 69.01 | 78.94 | 93.04 |
+--------+-------+-------+-------+

```
 **精度调试：**  
> 1.在线推理前处理图片是一定格式的动态分辨率，onnx将分辨率固定为512x512会导致精度下降些。  
> 2.分辨率在512x512时onnx离线推理的精度与om精度相同，分辨率改为500x500可以提升精度，使得mask的精度与开源相比更高  
> 3.单图调试  
> ```
> python3.7 mmsegmentation/tools/test.py mmsegmentation/configs/fcn/fcn_r50-d8_512x512_20k_voc12aug.py fcn_r50-d8_512x512_20k_voc12aug_20200617_010715-52dc5306.pth --show
> python3.7 mmsegmentation/tools/pytorch2onnx.py mmsegmentation/configs/fcn/fcn_r50-d8_512x512_20k_voc12aug.py --checkpoint fcn_r50-d8_512x512_20k_voc12aug_20200617_010715-52dc5306.pth --output-file fcn_r50-d8_512x512_20k_voc12aug.onnx --shape 500 500 --input-img 2011_003103.jpg --show --verify 
> ```


### 6.2 开源精度
[官网精度](https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r50-d8_512x512_20k_voc12aug/fcn_r50-d8_512x512_20k_voc12aug_20200617_010715.log.json)

```
{"mode": "val", "epoch": 31, "iter": 20000, "lr": 0.0001, "mIoU": 0.67085, "mAcc": 0.76958, "aAcc": 0.92709}
```
### 6.3 精度对比
om推理mIoU精度均为0.6901，开源mIoU精度为0.67085，om精度大于开源精度，精度达标  
 

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
1.benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
由于在线推理与onnx推理还不支持多batch，所以仅测om bs1，bs16的性能。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：
```
[e2e] throughputRate: 14.2564, latency: 101639
[data read] throughputRate: 24.7255, moduleLatency: 40.444
[preprocess] throughputRate: 22.102, moduleLatency: 45.2448
[infer] throughputRate: 14.3682, Interface throughputRate: 16.2017, moduleLatency: 69.2286
[post] throughputRate: 14.368, moduleLatency: 69.5993
```
Interface throughputRate: 16.2017，16.2017x4=64.8068即是batch1 310单卡吞吐率  

batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：
```
[[e2e] throughputRate: 13.459, latency: 107660
[data read] throughputRate: 23.5047, moduleLatency: 42.5446
[preprocess] throughputRate: 21.4117, moduleLatency: 46.7034
[infer] throughputRate: 13.5517, Interface throughputRate: 15.4271, moduleLatency: 73.3405
[post] throughputRate: 0.850975, moduleLatency: 1175.12
```
Interface throughputRate: 15.4271，15.4271x4=61.7084即是batch16 310单卡吞吐率

2.npu纯推理性能  
batch1的性能，执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果
```
./benchmark.x86_64 -round=20 -om_path=fcn_r50-d8_512x512_20k_voc12aug_bs1.om -device_id=0 -batch_size=1
```
PureInfer_perf_of_fcn_r50-d8_512x512_20k_voc12aug_bs1_in_device_0.txt:
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_fcn_r50-d8_512x512_20k_voc12aug_bs1_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 16.2574samples/s, ave_latency: 61.5162ms
----------------------------------------------------------------
```
batch16的性能，执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果
```
./benchmark.x86_64 -round=20 -om_path=fcn_r50-d8_512x512_20k_voc12aug_bs16.om -device_id=0 -batch_size=16
```
PureInfer_perf_of_fcn_r50-d8_512x512_20k_voc12aug_bs16_in_device_0.txt:
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_fcn_r50-d8_512x512_20k_voc12aug_bs16_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 15.5282samples/s, ave_latency: 64.4083ms
----------------------------------------------------------------
```

**性能优化：**  
> 没有遇到性能不达标的问题，故不需要进行性能优化

