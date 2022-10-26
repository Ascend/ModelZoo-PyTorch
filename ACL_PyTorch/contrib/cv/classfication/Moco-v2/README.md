# Moco-v2 ONNX模型端到端推理指导

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
	-   [6.2 TopN精度](#62-TopN精度)
	-   [6.3 精度对比](#63-精度对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[moco-v2论文](https://arxiv.org/abs/2003.04297)  

### 1.2 代码地址
[moco-v2代码](https://github.com/facebookresearch/moco)  
branch:master  
commit_id:78b69cafae80bc74cd1a89ac3fb365dc20d157d3
  
备注：commit_id是指基于该次提交时的模型代码做推理，通常选择稳定版本的最后一次提交，或代码仓最新的一次提交  

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN == 5.1.RC1
onnx == 1.7.0
pytorch == 1.5.0
torchvision == 0.6.0
```

### 2.2 python第三方库

```
numpy == 1.18.5
pillow == 7.2.0
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
[moco-v2预训练pth权重文件](https://www.hiascend.com/zh/software/modelzoo/detail/1/79c64b56e43642c3a2e62a84f9ed9897)  

**注意：**
> 预训练pth权重文件的文件名为model_lincls_best.pth.tar，无需修改，在后续流程中直接使用

2.执行pth2onnx脚本，生成onnx模型文件
```
python3 pthtar2onnx.py 1 model_lincls_best.pth.tar
```

"1": bs大小

"model_lincls_best.pth.tar": 输入的pth模型

修改bs大小来修改对应的输出onnx模型的命名；目前已通过测试的bs为1，4，8，16，32，64;

**注意：**
>此模型采用动态batchsize方式导出，存在精度和性能问题，暂无法规避，建议采用导出固定batchsize的方式

### 3.2 onnx转om模型

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23310P424%7C251366513%7C22892968%7C251168373)，需要指定输出节点以去除无用输出，可以使用netron开源可视化工具查看具体的输出节点名：

使用atc将onnx模型 ${chip_name}可通过npu-smi info指令查看

![img_1.png](https://images.gitee.com/uploads/images/2022/0713/160919_4b1ec998_8317185.png)

执行ATC命令

```shell
atc --model=moco-v2-bs1.onnx --framework=5 --output=moco-v2-atc-bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=info --soc_version=Ascend${chip_name} 
```
参数说明：
--model：为ONNX模型文件。 \
--framework：5代表ONNX模型。 \
--output：输出的OM模型。 \
--input_format：输入数据的格式。 \
--input_shape：输入数据的shape。 \
--log：日志级别。 \
--soc_version：处理器型号。 

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签存放路径分别为/path/dataset/ILSVRC2012_img_val/val与/path/val_label.txt

### 4.2 数据集预处理
1.预处理脚本imagenet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
mkdir -p prep_dataset
python3.7 imagenet_torch_preprocess.py ${datasets_path}/imageNet/val ./prep_dataset
```
“${datasets_path}/imageNet/val”：原始数据验证集（.jpeg）所在路径。

“./prep_dataset”：为输出的二进制文件（.bin）所在路径。每个图像对应生成一个二进制文件。
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```
python3.7 get_info.py bin ./prep_dataset ./dataset_prep_bin.info 224 224
```
“bin”：生成的数据集文件格式。

“./prep_dataset”：预处理后的数据文件的路径。

“./dataset_prep_bin.info”：生成的数据集文件保存的路径。
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310/310P上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理benchmark工具用户指南 01
### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理

```
chmod u+x benchmark.${arch}
./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=0 -input_text_path=./dataset_prep_bin.info -input_width=224 -input_height=224 -om_path=./moco-v2-atc-bs1.om -useDvpp=False -output_binary=False
```
推理后的输出默认在当前目录result下。

## 6 精度对比

-   **[离线推理TopN精度](#61-离线推理TopN精度)**  
-   **[TopN精度](#62-TopN精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理TopN精度统计

后处理统计TopN精度

调用vision_metric_ImageNet.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ ./val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  


### 6.2 TOPN精度

测试得到精度：
```
Model                  Acc@1        Acc@5
moco-v2_bs32_310	   67.31	    87.82
moco-v2_bs32_310P	   67.28	    87.82
```
### 6.3 精度对比
将得到的310om离线模型推理TopN精度与310P的精度对比，精度下降在1%范围之内，故精度达标。  

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device。为快速获取性能数据，也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准。这里给出两种方式，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
1.benchmark工具在整个数据集上推理获得性能数据  
性能表格为：
|         | 310      | 310P    | T4       | 310P/310   | 310P/T4  |
|---------|----------|---------|----------|------------|----------|
| bs1     | 1604.244 | 1371.17 | 915.818  | 0.854714   | 1.497208 |
| bs4     | 2172.888 | 3288.38 | 1313.518 | 1.513368   | 2.503371 |
| bs8     | 2418.532 | 3045.12 | 1066.401 | 1.259078   | 2.855511 |
| bs16    | 2444.788 | 2974.07 | 1692.353 | 1.216494   | 1.757358 |
| bs32    | 2237.228 | 2607.27 | 1734.916 | 1.165402   | 1.502822 |
| bs64    | 1703.216 | 2551.53 | 1870.913 | 1.498066   | 1.363789 |
| 最优batch | 2444.788 | 3288.38 | 1870.913 | 1.345057 | 1.757634 |

**性能优化：**  

>没有遇到性能不达标的问题，故不需要进行性能优化