# 基于开源mmsegmentation预训练的PSPnet Onnx模型端到端推理指导
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
	-   [7.2 T4性能数据](#72-T4性能数据)
	-   [7.3 性能对比](#73-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[PSPnet论文](https://arxiv.org/abs/1612.01105)  
论文使用PPM(pyramid pooling module)和提出的PSPNet(pyramid scene parsing network)，实现了通过融合different-region-based context获取全局context信息的能力。同时，PSPNet在多个数据集上实现了SOTA，取得ImageNet scene parsing challenge 2016、PASCAL VOC 2012 benchmark和Cityscapes benchmark的第1名。

### 1.2 代码地址
[mmsegmentation框架PSPnet代码](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet)   
branch:master commit_id:52b4fa5b9a3d65d0745d8bccb08ac0b88c9407fe

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
[PSPnet基于mmsegmentation预训练的npu权重文件](https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x512_20k_voc12aug/pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pth)  
文件md5sum: c563f7683bab2a869fe095a9eb801f6c
  
2.mmsegmentation源码安装
```shell
pip3.7 install mmcv-full==1.3.10
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
python3.7 mmsegmentation/tools/pytorch2onnx.py mmsegmentation/configs/pspnet/pspnet_r50-d8_512x512_20k_voc12aug.py --checkpoint pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pth --output-file pspnet_r50-d8_512x512_20k_voc12aug.onnx --shape 500 500  
```
4.通过onnx simplifier简化onnx模型  
```shell
python3.7 -m onnxsim  --input-shape="1,3,500,500" pspnet_r50-d8_512x512_20k_voc12aug.onnx pspnet_r50-d8_512x512_20k_voc12aug_sim.onnx
```
 **模型转换要点：**  
> 导出的onnx为固定batch1,不是动态batch。

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
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN 5.0.2 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)，如果存在多余输出节点,需要指定输出节点以去除无用输出，节点序号可能会因网络结构不同而不同，使用netron开源可视化工具查看具体的输出节点名：
生成bs1的om模型:
```shell
atc --framework=5 --model=pspnet_r50-d8_512x512_20k_voc12aug_sim.onnx  --output=pspnet_r50-d8_512x512_20k_voc12aug_sim_bs1 --input_format=NCHW --input_shape=" input:1,3,500,500" --log=debug --soc_version=Ascend310 --input_fp16_nodes=input
```
生成bs16的om模型:
```shell
atc --framework=5 --model=pspnet_r50-d8_512x512_20k_voc12aug_sim.onnx  --output=pspnet_r50-d8_512x512_20k_voc12aug_sim_bs16 --input_format=NCHW --input_shape=" input:16,3,500,500" --log=debug --soc_version=Ascend310 --input_fp16_nodes=input
```
 **模型转换要点：**  
> 通过input_fp16_nodes将输入的数据精度改为fp16,提升性能。

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

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN 5.0.2 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)

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
./benchmark.${arch} -model_type=vision -om_path=pspnet_r50-d8_512x512_20k_voc12aug_sim_fp16_bs1.om -device_id=0 -batch_size=1 -input_text_path=voc12.info -input_width=500 -input_height=500 -useDvpp=false -output_binary=true

./benchmark.${arch} -model_type=vision -om_path=pspnet_r50-d8_512x512_20k_voc12aug_sim_fp16_bs16.om -device_id=1 -batch_size=16 -input_text_path=voc12.info -input_width=500 -input_height=500 -useDvpp=false -output_binary=true
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
| background  | 93.78 | 97.28 |
| aeroplane   | 87.46 | 94.06 |
| bicycle     | 41.32 | 88.9  |
| bird        | 86.48 | 91.68 |
| boat        | 70.01 | 83.3  |
| bottle      | 76.2  | 84.19 |
| bus         | 92.78 | 96.14 |
| car         | 85.56 | 92.34 |
| cat         | 91.47 | 96.61 |
| chair       | 35.65 | 46.37 |
| cow         | 89.62 | 93.35 |
| diningtable | 55.73 | 59.82 |
| dog         | 86.24 | 92.88 |
| horse       | 88.84 | 93.02 |
| motorbike   | 83.75 | 92.17 |
| person      | 83.81 | 91.12 |
| pottedplant | 60.77 | 67.82 |
| sheep       | 87.55 | 91.34 |
| sofa        | 49.2  | 59.29 |
| train       | 85.96 | 91.59 |
| tvmonitor   | 67.55 | 79.11 |
+-------------+-------+-------+
Summary:

+--------+-------+-------+-------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+-------+
| global | 76.18 | 84.87 | 94.49 |
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
| background  | 93.78 | 97.28 |
| aeroplane   | 87.46 | 94.06 |
| bicycle     | 41.32 | 88.9  |
| bird        | 86.48 | 91.68 |
| boat        | 70.01 | 83.3  |
| bottle      | 76.2  | 84.19 |
| bus         | 92.78 | 96.14 |
| car         | 85.56 | 92.34 |
| cat         | 91.47 | 96.61 |
| chair       | 35.65 | 46.37 |
| cow         | 89.62 | 93.35 |
| diningtable | 55.73 | 59.82 |
| dog         | 86.24 | 92.88 |
| horse       | 88.84 | 93.02 |
| motorbike   | 83.75 | 92.17 |
| person      | 83.81 | 91.12 |
| pottedplant | 60.77 | 67.82 |
| sheep       | 87.55 | 91.34 |
| sofa        | 49.2  | 59.29 |
| train       | 85.96 | 91.59 |
| tvmonitor   | 67.55 | 79.11 |
+-------------+-------+-------+
Summary:

+--------+-------+-------+-------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+-------+
| global | 76.18 | 84.87 | 94.49 |
+--------+-------+-------+-------+
```
 **精度调试：**  
> 1.在线推理前处理图片是一定格式的动态分辨率，onnx将分辨率固定为512x512会导致精度下降些。  
> 2.分辨率在512x512时onnx离线推理的精度与om精度相同，分辨率改为500x500可以提升精度，使得mask的精度与开源相比更高  
> 3.单图调试  
> ```
> python3.7 mmsegmentation/tools/test.py mmsegmentation/configs/pspnet/pspnet_r50-d8_512x512_20k_voc12aug.py pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pth --show
> python3.7 mmsegmentation/tools/pytorch2onnx.py mmsegmentation/configs/pspnet/pspnet_r50-d8_512x512_20k_voc12aug.py --checkpoint pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pth --output-file pspnet_r50-d8_512x512_20k_voc12aug.onnx --shape 500 500 --input-img 2011_003103.jpg --show --verify 
> ```


### 6.2 开源精度
[官网精度](https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x512_20k_voc12aug/pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958.log.json)

```
{"mode": "val", "epoch": 31, "iter": 20000, "lr": 0.0001, "mIoU": 0.76778, "mAcc": 0.85529, "aAcc": 0.94787}
```
### 6.3 精度对比
om推理bs1和bs16的mIoU精度均为0.7618，开源mIoU精度为0.76778，om精度下降小于1%，精度达标  
 

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[T4性能数据](#72-T4性能数据)**  
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据
1.benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
由于在线推理与onnx推理还不支持多batch，所以仅测om bs1，bs16的性能。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：
```
[e2e] throughputRate: 7.85666, latency: 184430
[data read] throughputRate: 37.4296, moduleLatency: 26.7168
[preprocess] throughputRate: 28.1654, moduleLatency: 35.5045
[infer] throughputRate: 7.91227, Interface throughputRate: 8.19018, moduleLatency: 126.139
[post] throughputRate: 7.91221, moduleLatency: 126.387
```
Interface throughputRate: 7.91221，7.91221x4=31.64884即是batch1 310单卡吞吐率

batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：
```
[e2e] throughputRate: 8.16118, latency: 177548
[data read] throughputRate: 40.508, moduleLatency: 24.6865
[preprocess] throughputRate: 29.1145, moduleLatency: 34.3472
[infer] throughputRate: 8.21425, Interface throughputRate: 8.52684, moduleLatency: 121.508
[post] throughputRate: 0.515815, moduleLatency: 1938.68
```
Interface throughputRate: 8.21425，8.21425x4=32.857即是batch16 310单卡吞吐率

2.npu纯推理性能  
batch1的性能，执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果
```
./benchmark.x86_64 -round=20 -om_path=pspnet_r50-d8_512x512_20k_voc12aug_bs1.om -device_id=0 -batch_size=1
```
PureInfer_perf_of_pspnet_r50-d8_512x512_20k_voc12aug_bs1_in_device_0.txt:
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_pspnet_r50-d8_512x512_20k_voc12aug_bs1_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 8.12674samples/s, ave_latency: 123.129ms
----------------------------------------------------------------
```

batch6的性能，执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果
```
./benchmark.x86_64 -round=20 -om_path=pspnet_r50-d8_512x512_20k_voc12aug_bs16.om -device_id=0 -batch_size=16
```
PureInfer_perf_of_pspnet_r50-d8_512x512_20k_voc12aug_bs16_in_device_0.txt:
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_pspnet_r50-d8_512x512_20k_voc12aug_bs16_in_device_0.txt                                         
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 8.51957samples/s, ave_latency: 117.39ms
----------------------------------------------------------------
```
### 7.2 T4性能数据
在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2  
1.batch1性能：
```
trtexec --onnx=pspnet_r50-d8_512x512_20k_voc12aug_sim.onnx --fp16 --shapes=input:1,3,500,500 
```
gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch。其中--fp16是算子精度，目前算子精度只测--fp16的。注意--shapes是onnx的输入节点名与shape，当onnx输入节点的batch为-1时，可以用同一个onnx文件测不同batch的性能，否则用固定batch的onnx测不同batch的性能不准。
```
[09/24/2021-04:17:29] [I] GPU Compute
[09/24/2021-04:17:29] [I] min: 15.829 ms
[09/24/2021-04:17:29] [I] max: 20.5302 ms
[09/24/2021-04:17:29] [I] mean: 16.2649 ms
[09/24/2021-04:17:29] [I] median: 16.0951 ms
[09/24/2021-04:17:29] [I] percentile: 19.1857 ms at 99%
[09/24/2021-04:17:29] [I] total compute time: 3.04154 s

```
batch1 t4单卡吞吐率：1000/(16.2649/1)=61.482fps

2.batch16性能：
```
trtexec --onnx=pspnet_r50-d8_512x512_20k_voc12aug_sim.onnx --fp16 --shapes=input:16,3,500,500 
```
```
[09/24/2021-04:25:43] [I] GPU Compute
[09/24/2021-04:25:43] [I] min: 15.7839 ms
[09/24/2021-04:25:43] [I] max: 20.8466 ms
[09/24/2021-04:25:43] [I] mean: 16.2072 ms
[09/24/2021-04:25:43] [I] median: 16.0396 ms
[09/24/2021-04:25:43] [I] percentile: 19.1329 ms at 99%
[09/24/2021-04:25:43] [I] total compute time: 3.03074 s
```
batch16 t4单卡吞吐率：1000/(16.2072/1)=61.701fps

### 7.3 性能对比
batch1：7.91221x4 < 1000/(16.2649/1) 
batch1：8.21425x4 < 1000/(16.2072/1)   
310单个device的吞吐率乘4即单卡吞吐率比T4单卡的吞吐率小，故310性能低于T4性能，性能不达标。    

**性能优化：**  
> 由于onnx转om的过程中，两个avgpool算子的kernel size过大，导致被替换为aicpu算子，致使性能不足。需等优化底层算子后再进行测试。

