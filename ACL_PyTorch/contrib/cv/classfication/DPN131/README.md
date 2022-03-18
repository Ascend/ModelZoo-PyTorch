# DPN131 Onnx模型端到端推理指导
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

 

## 1  模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[DPN131论文](https://arxiv.org/abs/1707.01629)  
### 1.2 代码地址
[DPN131代码](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/dpn.py)  
branch: master
commit id : 0a4df4f3fe46b81e94bf9cc9ee5d9bebee6b9ec5


## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

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
opencv-python == 4.5.1.48
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
[DPN131预训练pth权重文件](http://data.lip6.fr/cadene/pretrainedmodels/dpn131-7af84be88.pth)  
文件md5sum: 71e7844aa8646dc75494976c7c86241a
```
wget http://data.lip6.fr/cadene/pretrainedmodels/dpn131-7af84be88.pth
```

2.安装过程如下所示：若安装过程报错请百度解决
```
git clone https://github.com/Cadene/pretrained-models.pytorch.git
cd ./pretrainedmodels/models/
patch -p1 < ../dpn.diff,其中dpn.diff是通过git diff > ./dpn.diff生成的
cd..
```
3.编写pth2onnx脚本dpn131_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 dpn131_pth2onnx.py ./dpn131-7af84be88.pth dpn131.onnx
```

 **模型转换要点：**  
>由于包含动态shape需要修改模型代码以固定shape


### 3.2 onnx转om模型

1.设置环境变量
```
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)
```
atc --framework=5 --model=./dpn131.onnx --output=dpn131_bs1 --input_format=NCHW --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend310

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
python3.7 imagenet_torch_preprocess.py dpn131 /root/datasets/imagenet/val ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 gen_dataset_info.py bin ./prep_dataset ./dpn131_prep_bin.info 224 224
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
source env.sh
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=dpn131_bs1.om -input_text_path=./dpn131_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
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
python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ /root/datasets/imagenet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：
```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "79.43%"}, {"key": "Top2 accuracy", "value": "88.87%"}, {"key": "Top3 accuracy", "value": "91.98%"}, {"key": "Top4 accuracy", "value": "93.57%"}, {"key": "Top5 accuracy", "value": "94.58%"}]}
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上

### 6.2 开源TopN精度
[DPN官网精度](https://github.com/rwightman/pytorch-dpn-pretrained)
```
Model        Acc@1     Acc@5
dpn131       79.432	   94.574
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 34.4577, latency: 1.45105e+06
[data read] throughputRate: 36.63, moduleLatency: 27.3
[preprocess] throughputRate: 36.5006, moduleLatency: 27.3968
[infer] throughputRate: 34.6141, Interface throughputRate: 37.1687, moduleLatency: 28.3242
[post] throughputRate: 34.614, moduleLatency: 28.89
```
Interface throughputRate: 37.1687，37.1687x4=148.6748既是batch1 310单卡吞吐率  

batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  
```
[e2e] throughputRate: 36.1546, latency: 1.38295e+06
[data read] throughputRate: 38.3744, moduleLatency: 26.059
[preprocess] throughputRate: 38.2771, moduleLatency: 26.1253
[infer] throughputRate: 36.2854, Interface throughputRate: 39.3897, moduleLatency: 26.8079
[post] throughputRate: 2.26783, moduleLatency: 440.951
```
Interface throughputRate: 39.3897，39.3897x4=157.5588既是batch16 310单卡吞吐率  
batch4性能：
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_dpn131_bs4_in_device_3.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 38.4344samples/s, ave_latency: 26.1589ms
----------------------------------------------------------------
```
batch4 310单卡吞吐率：38.4344x4=153.7376fps  
batch8性能：
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_dpn131_bs8_in_device_3.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 39.3665samples/s, ave_latency: 25.4697ms
----------------------------------------------------------------
```
batch8 310单卡吞吐率：39.3665x4=157.466fps  
batch32性能：
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_dpn131_bs32_in_device_3.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 40.1895samples/s, ave_latency: 24.9015ms
----------------------------------------------------------------
```
batch32 310单卡吞吐率：40.1895x4=160.758fps  

 **性能优化：**  
>待优化
蓝区社区版本CANN 5.0.2.alpha003优化了StridedSliceD，使用该版本测
sclice算子引入过多的transdata需要进一步优化


