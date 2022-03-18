# InceptionResNetV2 Onnx模型端到端推理指导
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
[InceptionResNetV2论文](https://arxiv.org/abs/1602.07261)  

### 1.2 代码地址
[InceptionResNetV2代码](https://github.com/Cadene/pretrained-models.pytorch#inception)  
branch:master  
commit id:3c92fbda001b6369968e7cb1a5706ee6bf6c9fd7  


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
numpy == 1.20.3
Pillow == 8.2.0
opencv-python == 4.5.2.54
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
[InceptionResNetV2pth权重文件](http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth)  
文件md5sum: 034a38b1e72c185cccf2e01a9ad458ac   

```
wget http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth

 ```
2.下载InceptionResNetV2源码
 ```
git clone https://github.com/Cadene/pretrained-models.pytorch
cd pretrained-models.pytorch
git reset commitid --hard
cd ..
如果使用补丁文件修改了模型代码则将补丁打入模型代码，如果需要引用模型代码仓的类或函数通过sys.path.append(r"./pretrained-models.pytorch")添加搜索路径。
```


3.编写pth2onnx脚本inceptionresnetv2_pth2onnx.py  

 **说明：**  

>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 inceptionresnetv2_pth2onnx.py inceptionresnetv2-520b38e4.pth inceptionresnetv2.onnx
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
atc --framework=5 --model=inceptionresnetv2.onnx --output=inceptionresnetv2-b0_bs1 --input_format=NCHW --input_shape="image:1,3,299,299" --log=debug --soc_version=Ascend310
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/root/datasets/imagenet/val与/root/datasets/imagenet/val_label.txt。

### 4.2 数据集预处理
1.预处理脚本imagenet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 imagenet_torch_preprocess.py inceptionresnetv2 /root/datasets/imagenet/val ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 gen_dataset_info.py bin ./prep_dataset ./inceptionresnetv2.info 299 299
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
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=inceptionresnetv2-b0_bs1.om -input_text_path=./inceptionresnetv2.info -input_width=299 -input_height=299 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_device{0}，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

后处理统计TopN精度

调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ /root/datasets/imagenet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：

```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1001"}, {"key": "Top1 accuracy", "value": "80.04%"}, {"key": "Top2 accuracy", "value": "89.5%"}, {"key": "Top3 accuracy", "value": "92.64%"}, {"key": "Top4 accuracy", "value": "94.17%"}, {"key": "Top5 accuracy", "value": "95.18%"}]}
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上  

### 6.2 开源精度
[精度](https://github.com/Cadene/pretrained-models.pytorch#inception)

```
Model               Acc@1     Acc@5
InceptionResNetV2    80.170    95.234
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
[e2e] throughputRate: 78.9555, latency: 633268
[data read] throughputRate: 81.6348, moduleLatency: 12.2497
[preprocess] throughputRate: 81.4266, moduleLatency: 12.281
[infer] throughputRate: 79.1538, Interface throughputRate: 104.996, moduleLatency: 12.4087
[post] throughputRate: 79.1537, moduleLatency: 12.6337
```
Interface throughputRate: 104.996，104.996x4=419.984即是batch1 310单卡吞吐率。


batch16的性能：
benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：

```
[e2e] throughputRate: 114.646, latency: 436127
[data read] throughputRate: 118.435, moduleLatency: 8.44347
[preprocess] throughputRate: 118.231, moduleLatency: 8.45803
[infer] throughputRate: 115.054, Interface throughputRate: 170.487, moduleLatency: 8.46028
[post] throughputRate: 7.19077, moduleLatency: 139.067
```
Interface throughputRate: 170.487，170.487x4=681.948即是batch16 310单卡吞吐率

batch4：

```
./benchmark.x86_64 -round=20 -om_path=inceptionresnetv2-b0_bs4.om -device_id=0 -batch_size=4
```

```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_inceptionresnetv2-b0_bs4_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 141.738samples/s, ave_latency: 7.35671ms
----------------------------------------------------------------
```
batch4 310单卡吞吐率：141.738x4=566.952fps  

batch8性能：
```
./benchmark.x86_64 -round=20 -om_path=inceptionresnetv2-b0_bs8.om -device_id=0 -batch_size=8
```

```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_inceptionresnetv2-b0_bs8_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 167.624samples/s, ave_latency: 6.14986ms
----------------------------------------------------------------
```
batch8 310单卡吞吐率：167.624x4=670.496fps 


batch32性能：

```
./benchmark.x86_64 -round=20 -om_path=inceptionresnetv2-b0_bs32.om -device_id=0 -batch_size=32
```
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_inceptionresnetv2-b0_bs32_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 173.11samples/s, ave_latency: 5.80854ms
----------------------------------------------------------------
```
batch32 310单卡吞吐率：173.11x4=692.44fps  

 **性能优化：**  

>从profiling看出主要耗时的算子是Conv2D，bs32的性能与基准基本持平



