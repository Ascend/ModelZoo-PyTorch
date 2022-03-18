# xception 模型端到端推理指导
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
[xception论文](https://arxiv.org/abs/1610.02357)  

### 1.2 代码地址
[xception代码](https://github.com/tstandley/Xception-PyTorch)  

branch:master  
commit id:7b9718bb525fefc95f507306e685aa8998d0492c




### 2.1 深度学习框架
```
CANN 5.0.1

pytorch >= 1.5.0
torchvision >= 0.6.0
onnx >= 1.7.0
```

### 2.2 python第三方库

```
numpy == 1.18.5
Pillow == 7.2.0
opencv-python == 4.2.0.34
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
xception预训练pth权重文件
 
```
https://ascend-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/image_classification/xception/pth/xception-c0a72b38.pth.tar  
```
文件md5sum:203a36dce3e49d45e5d742efbad5ba65 


2.下载xception源码：  
```
git clone https://github.com/tstandley/Xception-PyTorch
cd Xception-PyTorch  
git reset 7b9718bb525fefc95f507306e685aa8998d0492c --hard  
cd ..
```
如果使用补丁文件修改了模型代码则将补丁打入模型代码，如果需要引用模型代码仓的类或函数通过sys.path.append(r"./Xception-PyTorch")添加搜索路径。

3.编写pth2onnx脚本xception_pth2onnx.py  
 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 xception_pth2onnx.py  xception-c0a72b38.pth.tar  xception.onnx
```

 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明

### 3.2 onnx转om模型

1.设置环境变量
```
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01
```
atc --framework=5 --model=xception.onnx --output=xception_1 --input_format=NCHW --input_shape="image:1,3,299,299" --log=debug --soc_version=Ascend310 
```
 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明




## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/root/datasets/imagenet/val与/root/datasets/imagenet/val_label.txt。 

### 4.2 数据集预处理
1.预处理脚本img_preprocess.py


2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 img_preprocess.py /opt/npu/imagenet/val ./pre_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 gen_dataset_info.py bin ./prepro_dataset ./xception.info 299 299
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息





## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述
benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理benchmark工具用户指南 01
### 5.2 离线推理
1.设置环境变量
```
source env.sh
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=xception.om -input_text_path=./xception.info -input_width=299 -input_height=299 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_device{0}，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。



## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[开源TopN精度](#62-开源TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

后处理统计TopN精度

调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python imagenet_acc_eval.py result/dumpOutput_device0/ /root/datasets/imagenet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：
```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "78.35%"}, {"key": "Top2 accuracy", "value": "88.11%"}, {"key": "Top3 accuracy", "value": "91.39%"}, {"key": "Top4 accuracy", "value": "93.16%"}, {"key": "Top5 accuracy", "value": "94.31%"}]}
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上。

### 6.2 开源TopN精度
  
```
torchvision官网精度

Model               Acc@1     Acc@5
Xception            78.892    94.292   
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。
精度调试：
没有遇到精度不达标的问题，故不需要进行精度调试




## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。
1.benchmark工具在整个数据集上推理获得性能数据
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：
```

[e2e] throughputRate: 108.021, latency: 462872
[data read] throughputRate: 111.695, moduleLatency: 8.95293
[preprocess] throughputRate: 111.375, moduleLatency: 8.97865
[infer] throughputRate: 108.345, Interface throughputRate: 158.623, moduleLatency: 9.01342
[post] throughputRate: 108.345, moduleLatency: 9.22979
```
Interface throughputRate: 158.623，158.623x4=634.492既是batch1 310单卡吞吐率

batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：
```
[e2e] throughputRate: 127.923, latency: 390859
[data read] throughputRate: 131.925, moduleLatency: 7.58008
[preprocess] throughputRate: 131.775, moduleLatency: 7.5887
[infer] throughputRate: 128.194, Interface throughputRate: 198.865, moduleLatency: 7.58156
[post] throughputRate: 8.012, moduleLatency: 124.813
```

Interface throughputRate: 198.865，198.865x4=795.46既是batch16 310单卡吞吐率
batch4性能：
```
[e2e] throughputRate: 119.972, latency: 416763
[data read] throughputRate: 124.007, moduleLatency: 8.06404
[preprocess] throughputRate: 123.625, moduleLatency: 8.08897
[infer] throughputRate: 120.284, Interface throughputRate: 182.215, moduleLatency: 8.10143
[post] throughputRate: 30.071, moduleLatency: 33.2546
```
batch4 310单卡吞吐率：182.215x4=728.86fps
batch8性能：
```
[e2e] throughputRate: 129.279, latency: 386759
[data read] throughputRate: 133.583, moduleLatency: 7.486
[preprocess] throughputRate: 133.456, moduleLatency: 7.49308
[infer] throughputRate: 129.749, Interface throughputRate: 203.357, moduleLatency: 7.49475
[post] throughputRate: 16.2184, moduleLatency: 61.6582
```

batch8 310单卡吞吐率：203.357x4=813.428fps


batch32性能：
```
[e2e] throughputRate: 112.895, latency: 442890
[data read] throughputRate: 116.539, moduleLatency: 8.5808
[preprocess] throughputRate: 116.412, moduleLatency: 8.59021
[infer] throughputRate: 113.12, Interface throughputRate: 164.302, moduleLatency: 8.62059
[post] throughputRate: 3.53608, moduleLatency: 282.799
```
batch32 310单卡吞吐率：164.302x4=657.208fps

 **性能优化：**  


batch32纯推理性能达标


