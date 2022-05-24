# Mnasnet1.0 Onnx模型端到端推理指导
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
	-   [7.2 T4性能数据](#72-T4性能数据)
	-   [7.3 性能对比](#73-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[Mnasnet论文](https://arxiv.org/pdf/1807.11626.pdf)  

### 1.2 代码地址
[Mnasnet代码](https://github.com/pytorch/vision/blob/master/torchvision/models/mnasnet.py)  
branch:master
commit id:91e03b91fd9bab19b4c295692455a1883831a932
## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

### 2.1 深度学习框架
```
CANN 5.0.1

torch == 1.5.0
torchvision == 0.6.0
onnx == 1.7.0

pip install torch==1.5.0 torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install onnx=1.7.0
```

### 2.2 python第三方库

```
numpy == 1.20.3
Pillow == 8.2.0
opencv-python == 4.5.2.54

pip install numpy==1.20.3
pip install Pillow==8.2.0
pip install nopencv-python==4.5.2.54
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
[Mnasnet1.0预训练pth权重文件](https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth)  
```
wget https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth
```
文件MD5sum：02d9eb9b304e14cfe0e7ea057be465f0

2.Mnasnet1.0模型代码在torchvision里，安装torchvision，arm下需源码安装，参考torchvision官网，若安装过程报错请百度解决
```
git clone https://github.com/pytorch/vision
cd vision
python3.7 setup.py install
cd ..
```
3.编写pth2onnx脚本mnasnet_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python mnasnet_pth2onnx.py ./mnasnet1.0_top1_73.512-f206786ef8.pth mnasnet1.0.onnx
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
./onnx2om.sh
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取

userDatasetPath=/home/zhx/datasets/imagenet

该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在 $userDatasetPath/val 与$userDatasetPath/val_label.txt。

### 4.2 数据集预处理
1.预处理脚本imagenet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python imagenet_torch_preprocess.py mnasnet $userDatasetPath/val ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```
python gen_dataset_info.py bin ./prep_dataset ./mnasnet_prep_bin.info 224 224
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)

msame：https://gitee.com/ascend/tools/tree/master/msame

### 5.2 离线推理
1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
```
chmod 777 benchmark.sh
./benchmark.sh
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
python imagenet_acc_eval.py result/dumpOutput_device0/ $userDatasetPath/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：

```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "73.4%"}, {"key": "Top2 accuracy", "value": "83.9%"}, {"key": "Top3 accuracy", "value": "87.87%"}, {"key": "Top4 accuracy", "value": "90.08%"}, {"key": "Top5 accuracy", "value": "91.5%"}]}
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上

### 6.2 开源TopN精度
[torchvision官网精度](https://pytorch.org/vision/stable/models.html)
```
Model        Acc@1     Acc@5
MNASNet1.0  73.456    91.510
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  
-   **[T4性能数据](#72-T4性能数据)**  
-   **[性能对比](#73-性能对比)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  

Interface throughputRate:2113.01，是batch1 710单卡吞吐率
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  

Interface throughputRate: 9836.44，是batch16 710单卡吞吐率  
batch4性能：

batch4 710单卡吞吐率：6606.25fps  
batch8性能：

batch8 710单卡吞吐率：8926.07fps  
batch32性能：

batch32 710单卡吞吐率：10649.9fps

batch64性能：

batch64 710单卡吞吐率：5511.77fps

### 7.2 T4性能数据
在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2  
batch1性能：
```
trtexec --onnx=mnasnet1.0.onnx --fp16 --shapes=image:1x3x224x224 --threads
```
gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch

batch1 t4单卡吞吐率：1000x1/( 0.534034 /1)=1872.539fps  
batch16性能：

```
trtexec --onnx=mnasnet1.0.onnx --fp16 --shapes=image:16x3x224x224 --threads
```
batch16 t4单卡吞吐率：1000x1/(3.43677/16)=4655.534fps

batch4性能：

batch4 t4单卡吞吐率：1000x1/( 0.99884/4)= 4004.645fps

batch8性能：

batch8 t4单卡吞吐率：1000x1/( 1.72644/8)= 4633.813fps

batch32性能：

batch32 t4单卡吞吐率：1000x1/( 6.82446/32)= 4689.016fps

batch64性能：

batch32 t4单卡吞吐率：1000x1/( 13.6909/64)= 4674.638fps

### 7.3 性能对比
batch1：2113.01 > 1000x1/(0.534034/1) =1872.539
batch16：9836.44> 1000x1/(3.43677/16)=4655.534

| batchsize | 310/FPS    | 710/FPS    | T4/FPS      | 710/310 | 710/T4  |
| --------- | ---------- | ---------- | ----------- | ------- | ------- |
| 1         | 3106.844   | 2113.01    | 1872.54     | 0.68011 | 1.12842 |
| 4         | 5682.52    | 6606.25    | 4004.645    | 1.16256 | 1.64965 |
| 8         | 5868.36    | 8926.09    | 4633.818    | 1.52105 | 1.92639 |
| 16        | 6005.48    | 9836.44    | 4655.534    | 1.63791 | 2.11285 |
| 32        | 5590.4     | 10649.9    | 4689.016    | 1.90503 | 2.27124 |
| 64        | 5293.56    | 5511.77    | 4674.638    | 1.04122 | 1.17908 |
| 最优      | 16:6005.48 | 32:10649.9 | 32:4689.016 | 1.77336 | 2.27125 |

npu的吞吐率乘4比T4的吞吐率大，也等同于npu的时延除4比T4的时延除以batch小，故npu性能高于T4性能，性能达标。  
对于batch1与batch16，npu性能均高于T4性能1.2倍，该模型放在Benchmark/cv/classification目录下。  
**性能优化：**  

>没有遇到性能不达标的问题，故不需要进行性能优化
