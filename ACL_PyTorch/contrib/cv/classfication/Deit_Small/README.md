# Deit-small Onnx模型端到端推理指导

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

[Deit论文](https://arxiv.org/abs/2012.12877)  

### 1.2 代码地址

[Deit代码](https://github.com/facebookresearch/deit)  
branch:main  
commit id:6fa7ef60b4144b1e78f2cdb05598dce950e16ba6  

  


## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架

```
CANN 5.1.RC1

pytorch == 1.8.0
torchvision == 0.9.0
onnx == 1.7.0
onnx-simplifier == 0.3.6
```
pytorch==1.5.0时 使用onnxsim对网络进行优化会报错
### 2.2 python第三方库

```
numpy == 1.21.1
Pillow == 8.2.0
opencv-python == 4.5.2.54
timm == 0.3.2
```


**说明：** 

>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型 

1.clone官方仓代码

```bash
git clone https://github.com/facebookresearch/deit.git
```

2.下载pth权重文件  
[Deit-small预训练pth权重文件](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)  

```bash
wget https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth
```

3.编写pth2onnx脚本deit_small_pth2onnx.py

 **说明：**  

>- 注意目前ATC支持的onnx算子版本为11
>- 由于Deit未提供安装脚本，须在脚本中引用官方库作为类

4.执行pth2onnx脚本，生成onnx模型文件

```bash
python3.7 deit_small_pth2onnx.py deit_small_patch16_224-cd65a155.pth deit_small_patch16_224_onnx.onnx
```
其中第一个参数为deit权重，第二个参数为输出的onnx模型名称

onnxsim对网络进行优化

```bash
python3.7 -m onnxsim --input-shape="8,3,224,224" deit_small_patch16_224_onnx.onnx deit_bs8.onnx

```

 **模型转换要点：**  
onnx含有的where动态shape算子可以通过onnxsimplifier转换为静态shape优化掉，因此动态batch的onnx转om失败并且测的性能数据也不对，每个batch的om都需要对应batch的onnx来转换，每个batch的性能数据也需要对应batch的onnx来测

5.安装magic_onnx，优化模型
[install magiconnx, download url](https://gitee.com/Ronnie_zheng/MagicONNX/tree/refactor/)

```bash
git clone https://gitee.com/Ronnie_zheng/MagicONNX.git -b refactor
cd MagicONNX
pip3 install .
```
把deit_model.py移动到MagicONNX文件夹中，执行deit_model.py
```bash
python3 deit_model.py ./deit_bs8.onnx ./deit_magic_bs8.onnx
```


### 3.2 onnx转om模型

1.设置环境变量，注意请以实际安装环境配置环境变量。


```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN 5.0.1开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)

```bash
atc --framework=5 --model=/MagicONNX/deit_magic_bs8.onnx --output=deit_bs8 --input_format=NCHW --input_shape="image:8,3,224,224" --log=debug --soc_version=Ascend{$chip_name}
```
chip_name通过如下命令查看：
```bash
npu-smi info
```


## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取

该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/opu/npu/ILSVRC2012/val与/opu/npu/ILSVRC2012/val_label.txt。

### 4.2 数据集预处理

1.预处理脚本imagenet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件

```bash
python3.7 imagenet_torch_preprocess.py deit /opu/npu/ILSVRC2012/val ./prep_dataset
```

### 4.3 生成数据集信息文件

1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```
python3.7 gen_dataset_info.py bin ./prep_dataset ./deit_prep_bin.info 224 224
```

第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息

## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在处理器上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.1.RC1 推理benchmark工具用户指南 01

### 5.2 离线推理

1.设置环境变量，注意请以实际安装环境配置环境变量。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.增加benchmark.{arch}可执行权限

```bash
chmod u+x benchmark.x86_64
```

3.执行离线推理

```bash
sudo ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=8 -om_path=deit_bs8.om -input_text_path=./deit_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```

2.执行离线推理


输出结果默认保存在当前目录result/dumpOutput_device{0}，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[离线推理精度](#61-离线推理精度)**  
-   **[开源精度](#62-开源精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度统计

后处理统计TopN精度

调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。

```bash
python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ /opt/npu/val_label.txt ./ result.json
```

第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  

输出结果(310p)：

| Om Model | Acc@1 | Acc@5 |
| -------- | ----- | ----- |
| BS1      | 79.5 | 94.83 |
| BS8      | 79.5 | 94.83 |
| BS32     | 79.5 | 94.83 |
| Official | 79.9  | 95.0  |


```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "79.69%"}, {"key": "Top2 accuracy", "value": "89.22%"}, {"key": "Top3 accuracy", "value": "92.39%"}, {"key": "Top4 accuracy", "value": "93.98%"}, {"key": "Top5 accuracy", "value": "94.97%"}]}
```

经过对bs1与bs8的om测试，本模型batch1的精度与batch8的精度没有差别，精度数据均如上  
6.2 开源精度

[Deit官方精度](https://github.com/facebookresearch/deit)

```
Model               Acc@1     Acc@5
Deit-small          79.9      95.0
```

### 6.3 精度对比

将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  

>没有遇到精度不达标的问题，故不需要进行精度调试  

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  


### 7.1 npu性能数据


benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs8的性能数据为准，对于使用benchmark工具测试的batch4，16，32的性能数据在README.md中如下作记录即可。  


1.benchmark工具在整个数据集上推理获得性能数据  

batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  


[e2e] throughputRate: 87.7802, latency: 569604
[data read] throughputRate: 92.9318, moduleLatency: 10.7606
[preprocess] throughputRate: 92.6819, moduleLatency: 10.7896
[inference] throughputRate: 88.0032, Interface throughputRate: 103.795, moduleLatency: 10.7968
[postprocess] throughputRate: 88.0046, moduleLatency: 11.363
```

Interface throughputRate: 103.795，103.795x4=415.18既是batch1 310单卡吞吐率  



batch8的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_8_device_0.txt：  

```
[e2e] throughputRate: 93.9502, latency: 532197
[data read] throughputRate: 100.308, moduleLatency: 9.96932
[preprocess] throughputRate: 100.228, moduleLatency: 9.97721
[inference] throughputRate: 95.0294, Interface throughputRate: 122.054, moduleLatency: 9.64742
[postprocess] throughputRate: 11.8803, moduleLatency: 84.173
```


**性能优化：**  



>从profiling性能数据看出，TransData，SoftmaxV2，BatchMatMulV2耗时占比最高，其中SoftmaxV2算子存在对最后一维操作时性能低的问题，尝试通过转轴优化，TransData与BatchMatMulV2可以继续优化

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md