# HRNet Onnx模型端到端推理指导
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
	-   [6.1 离线推理TopN精度统计](#61-离线推理TopN精度统计)
	-   [6.2 开源TopN精度](#62-开源TopN精度)
	-   [6.3 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[HRNet论文](https://arxiv.org/pdf/1908.07919.pdf)  
Abstract—High-resolution representations are essential for position-sensitive vision problems, such as human pose estimation, semantic segmentation, and object detection. Existing state-of-the-art frameworks first encode the input image as a low-resolution representation through a subnetwork that is formed by connecting high-to-low resolution convolutions in series (e.g., ResNet, VGGNet), and then recover the high-resolution representation from the encoded low-resolution representation. Instead, our proposed network, named as High-Resolution Network (HRNet), maintains high-resolution representations through the whole process. There are two key characteristics: (i) Connect the high-to-low resolution convolution streams in parallel; (ii) Repeatedly exchange the information across resolutions. The benefit is that the resulting representation is semantically richer and spatially more precise. We show the superiority of the proposed HRNet in a wide range of applications, including human pose estimation, semantic segmentation, and object detection, suggesting that the HRNet is a stronger backbone for computer vision problems. All the codes are available at https://github.com/HRNet.

### 1.2 代码地址
[HRNet代码](https://github.com/HRNet/HRNet-Image-Classification)  
branch:master  
commit_id:f130a24bf09b7f23ebd0075271f76c4a188093b2  

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
pytorch >= 1.5.0
torchvision >= 0.6.0
onnx >= 1.7.0
```

### 2.2 python第三方库

```
numpy == 1.20.2
opencv-python == 4.5.2.52
Pillow == 8.0.1
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.下载HRNet模型  
git clone https://github.com/HRNet/HRNet-Image-Classification.git  

2.获取权重文件  
wget https://ascend-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/image_classification/HRNet/NPU/8P/model_best.pth.tar  
file name:model_best.pth.tar  
md5sum:1f1d61e242ac9ca4cab5d0c49299cb76  
  
3.编写pth2onnx脚本hrnet_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 hrnet_pth2onnx.py --cfg ./HRNet-Image-Classification/experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --input model_best.pth.tar --output hrnet_w18.onnx
```

 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明

### 3.2 onnx转om模型

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)

```
atc --framework=5 --model=./hrnet_w18.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=hrnet_bs16 --log=debug --soc_version=Ascend310
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/opt/npu/imagenet/val与/opt/npu/imagenet/val_label.txt。

### 4.2 数据集预处理
1.预处理脚本imagenet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件

```
datasets_path = '/opt/npu/'
python3.7 imagenet_torch_preprocess.py hrnet ${datasets_path}/imagenet/val ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```
python3.7 get_info.py bin ./prep_dataset ./hrnet_prep_bin.info 224 224
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)
### 5.2 离线推理
1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=hrnet_bs16.om -input_text_path=./hrnet_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_devicex，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

后处理统计TopN精度

调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ ${datasets_path}/imagenet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：

```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "76.46%"}, {"key": "Top2 accuracy", "value": "86.33%"}, {"key": "Top3 accuracy", "value": "90.0%"}, {"key": "Top4 accuracy", "value": "91.97%"}, {"key": "Top5 accuracy", "value": "93.14%"}]}
```
batch1,batch16的精度相同如上.

### 6.2 开源TopN精度
[torchvision官网精度](https://pytorch.org/vision/stable/models.html)
```
Model                         Acc@1     Acc@5
HRNet-Image-Classification    76.8      93.4
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
需要测试batch1，batch4，batch8，batch16，batch32的性能，这里用batch1与batch16做示例  
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device。为快速获取性能数据，也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准。这里给出两种方式，模型的测试脚本使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准。  
1.benchmark工具在整个数据集上推理获得性能数据  
以batch1为例，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  

```
[e2e] throughputRate: 125.08, latency: 399743
[data read] throughputRate: 132.686, moduleLatency: 7.53661
[preprocess] throughputRate: 132.548, moduleLatency: 7.54441
[infer] throughputRate: 125.448, Interface throughputRate: 156.216, moduleLatency: 7.35352
[post] throughputRate: 125.448, moduleLatency: 7.97142
```
Interface throughputRate: 156.216，156.216乘以4既是310单卡吞吐率  
2.benchmark纯推理功能测得性能数据  
batch1的性能：
 测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务

```
./benchmark.x86_64 -round=20 -om_path=hrnet_bs1.om -device_id=0 -batch_size=1
```
执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果
```
[INFO] Dataset number: 19 finished cost 2.635ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_hrnet_bs1_in_device_0.txt
-----------------PureInfer Performance Summary------------------
ave_throughputRate = 155.975samples/s, ave_latency = 6.42435ms
```
bs1 310单卡吞吐率:155.975x4=623.9fps/card  
batch4的性能：
```
./benchmark.x86_64 -round=20 -om_path=hrnet_w18_bs4.om -device_id=0 -batch_size=4
```
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_hrnet_w18_bs4_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 224.608samples/s, ave_latency: 4.56663ms
```
bs4 310单卡吞吐率:224.608x4=898.432fps/card    
batch8的性能：
```
./benchmark.x86_64 -round=20 -om_path=hrnet_w18_bs8.om -device_id=0 -batch_size=8
```
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_hrnet_w18_bs8_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 248.514samples/s, ave_latency: 4.09695ms
```
bs8 310单卡吞吐率:248.514x4=994.056fps/card  
batch16的性能：

```
./benchmark.x86_64 -round=20 -om_path=hrnet_w18_bs16.om -device_id=0 -batch_size=16
```
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_hrnet_w18_bs16_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 269.512samples/s, ave_latency: 3.73541ms
```
bs16 310单卡吞吐率:269.512x4=1078.048fps/card  
batch32的性能:
```
./benchmark.x86_64 -round=20 -om_path=hrnet_w18_bs32.om -device_id=0 -batch_size=32
```
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_hrnet_w18_bs32_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 271.562samples/s, ave_latency: 3.69597ms
```
bs32 310单卡吞吐率:271.562x4=1086.248fps/card  

 **性能优化：**  

>没有遇到性能不达标的问题，故不需要进行性能优化

