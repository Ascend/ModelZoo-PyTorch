# Dino_Resnet50 Onnx模型PyTorch离线推理指导
- [Dino_Resnet50 Onnx模型PyTorch离线推理指导](#Dino_Resnet50Onnx模型Pytorch离线推理指导)
	- [1 模型概述](#1-模型概述)
		- [1.1 论文地址](#11-论文地址)
		-   [1.2 代码地址](#12-代码地址)
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
		- [5.1 benchmark工具概述](#51-benchmark工具概述)
		- [5.2 离线推理](#52-离线推理)
	- [6 精度对比](#6-精度对比)
		- [6.1 310离线推理TopN精度统计](#61-310离线推理topn精度统计)
		- [6.2 310P离线推理TopN精度统计](#62-310p离线推理topn精度统计)
		- [6.3 开源精度](#63-开源精度)
		- [6.4 精度对比](#64-精度对比)
	- [7 性能对比](#7-性能对比)
		- [7.1 310性能数据](#71-310性能数据)
		- [7.2 310P性能数据](#72-310p性能数据)
		- [7.3 T4性能数据](#73-t4性能数据)
		- [7.4 性能对比](#74-性能对比)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  
-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[Caron, Mathilde, et al. "Emerging properties in self-supervised vision transformers." arXiv preprint arXiv:2104.14294 (2021).](https://arxiv.org/abs/2104.14294)

### 1.2 代码地址
[Dino代码地址](https://github.com/facebookresearch/dino)  
branch:main  
commit_id:cb711401860da580817918b9167ed73e3eef3dcf  
备注：commit_id是指基于该次提交时的模型代码做推理，通常选择稳定版本的最后一次提交，或代码仓最新的一次提交  

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  
-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC1
pytorch >= 1.5.0
torchvision >= 0.6.0
onnx >= 1.7.0
```

### 2.2 python第三方库

```
numpy == 1.18.5
Pillow == 7.2.0
opencv-python == 3.3.1
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  
-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.获取权重文件，从源码包中获取权重文件：dino_resnet50_pretrain.pth 和 dino_resnet50_linearweights.pth。

2.编写pthtar2onnx脚本dino_resnet50_pth2onnx.py

3.执行dino_resnet50_pth2onnx脚本，生成onnx模型文件
```
python3.7 dino_resnet50_pth2onnx.py
```

 **模型转换要点：**
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明

### 3.2 onnx转om模型

1.设置环境变量
```
source /usr/local/Ascend/ascend-lastest/set_env.sh
```
2.使用atc将onnx模型转换为om模型文件，${chip_name}可通过npu-smi info指令查看，例：310P3

${chip_name}可通过`npu-smi info`指令查看

![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```
atc --framework=5 --model=dino_resnet50.onnx --output=dino_resnet50_bs1 --input_format=NCHW --input_shape="input:1,3,224,224" --log=debug --soc_version=Ascend${chip_name} 
```

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  
-   **[数据集预处理](#42-数据集预处理)**  
-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，以ILSVRC2012为例，本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。

### 4.2 数据集预处理
1.编写预处理脚本dino_resnet50_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 dino_resnet50_preprocess.py dino ${datasets_path}/val ./${prep_output_dir}
```
resnet表示数据预处理方式为dino网络，{datasets_path}/val表示原始数据验证集（.jpeg）所在路径，${prep_output_dir}表示输出的二进制文件（.bin）所在路径。

### 4.3 生成数据集信息文件
1.编写数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 get_info.py bin ./${prep_output_dir} ./${prep_output_dir_info} 224 224
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息

## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  
-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)
### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-lastest/set_env.sh
```
2.执行离线推理
```
 ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./dino_resnet50_bs1.om -input_text_path=./dino_val.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_device{0}，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[310离线推理TopN精度](#61-310离线推理TopN精度)**
-   **[310P离线推理TopN精度](#62-310P离线推理TopN精度)** 
-   **[开源精度](#63-开源精度)**  
-   **[精度对比](#64-精度对比)**  

### 6.1 310离线推理TopN精度统计

后处理统计TopN精度

执行dino_resnet50_postprocess.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 dino_resnet50_postprocess.py --anno_file ${datasets_path}/val_label.txt --benchmark_out ./result/dumpOutput_device0 --result_file ./result.json
```

--benchmark_out表示生成推理结果所在路径，--anno_file表示标签数据，--result_file表示生成结果文件

310精度结果：
|         | Top1 accuracy | Top5 accuracy |
|---------|---------------|---------------|
| 310     | 75.27%        | 92.57%        |

### 6.2 310P离线推理TopN精度统计

同310，执行dino_resnet50_postprocess.py脚本：
```
python3.7 dino_resnet50_postprocess.py --anno_file ${datasets_path}/val_label.txt --benchmark_out ./result/dumpOutput_device0 --result_file ./result.json
```

310P精度结果：
|         | Top1 accuracy | Top5 accuracy |
|---------|---------------|---------------|
| 310P    | 75.28%        | 92.57%        |

### 6.3 开源精度

[官网pth精度](https://github.com/facebookresearch/dino#evaluation-linear-classification-on-imagenet)

| model            | Top1 accuracy   |
| ---------------- | --------------- |
| Dino_Resnet50    | 75.3%           |

### 6.4 精度对比

将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度不低于开源代码仓精度的1%，故精度达标。

 **精度调试：**  

>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[310性能数据](#71-310性能数据)**
-   **[310P性能数据](#72-310P性能数据)**  
-   **[T4性能数据](#73-T4性能数据)**  
-   **[性能对比](#74-性能对比)**  

### 7.1 310性能数据

benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。

使用benchmark工具在整个数据集上推理获得性能数据，可以获得吞吐率数据，结果保存在当前目录result/dumpOutput_device{0}。

```
 ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./dino_resnet50_bs1.om -input_text_path=./dino_val.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```

### 7.2 310P性能数据

同310，使用benchmark工具在整个数据集上推理获得性能数据：
```
 ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./dino_resnet50_bs1.om -input_text_path=./dino_val.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```

### 7.3 T4性能数据

在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务。

使用benchmark工具在整个数据集上推理获得性能数据：
```
trtexec  --onnx=dino_resnet50.onnx  --fp16 --shapes=input:1x3x224x224 --workspace=5000 --threads
```

### 7.4 性能对比

性能对比表格如下：

|           | 310      | 310P    | T4        | 310P/310    | 310P/T4     |
| --------- | -------- | ------- | --------- | ----------- | ----------- |
| bs1       | 1617.052 | 1378.5  | 878.742   | 0.852477224 | 1.568719829 |
| bs4       | 2161.044 | 3317.4  | 1532.616  | 1.535091373 | 2.164534365 |
| bs8       | 2410.052 | 3678.53 | 1733.528  | 1.526328063 | 2.12199053  |
| bs16      | 2441.26  | 2708.79 | 1858.176  | 1.109586853 | 1.457768263 |
| bs32      | 5279.8   | 2579.54 | 2033.92   | 1.1491282   | 1.268260305 |
| bs64      | 2244.78  | 4248.18 | 2090.9376 | 2.492466593 | 2.031710559 |
| 最优batch | 2441.26  | 4248.18 | 2090.9376 | 1.74015877  | 2.031710559 |

最优的310P性能达到了最优的310性能的1.740倍，达到最优的T4性能的2.031倍。

**性能优化：**
性能已达标，不需要再优化。