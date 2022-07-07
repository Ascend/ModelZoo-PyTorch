# spnasnet_net Onnx模型端到端推理指导
- [spnasnet_net Onnx模型端到端推理指导](#spnasnet_net-onnx模型端到端推理指导)
	- [1 模型概述](#1-模型概述)
		- [1.1 论文地址](#11-论文地址)
		- [1.2 代码地址](#12-代码地址)
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
		- [6.3 开源TopN精度](#63-开源topn精度)
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
[spnasnet_100论文](https://arxiv.org/abs/1904.02877)  

### 1.2 代码地址
[spnasnet_100代码](https://github.com/rwightman/pytorch-image-models)  
branch:master  
commit_id:54a6cca27a9a3e092a07457f5d56709da56e3cf5  
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
[spnasnet_100预训练pth权重文件](https://ascend-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/image_classification/spnasnet_100/NPU/8P/model_best.pth.tar)  
文件md5sum: 28bdc27f4fdbfe066c95f36b72417340
- 备注：本模型所使用权重文件由npu训练提供
```
wget https://ascend-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/image_classification/spnasnet_100/NPU/8P/model_best.pth.tar
```
2.spnasnet_100模型代码在timm仓里，安装timm，源码安装，若安装过程报错请百度解决
```
pip3.7 install git+https://github.com/rwightman/pytorch-image-models.git
```
3.编写pthtar2onnx脚本pthtar2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pthtar2onnx脚本，生成onnx模型文件
```
python3.7 pthtar2onnx.py model_best.pth.tar spnasnet_100.onnx
```

 **模型转换要点：**
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明

### 3.2 onnx转om模型

1.设置环境变量
```
source /usr/local/Ascend/ascend-lastest/set_env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)

${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```
atc --model=./spnasnet_100.onnx --framework=5 --output=spnasnet_100_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=info --soc_version=Ascend${chip_name}
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  
-   **[数据集预处理](#42-数据集预处理)**  
-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/root/datasets/imagenet/val与/root/datasets/imagenet/val_label.txt。

### 4.2 数据集预处理
1.预处理脚本preprocess_spnasnet_100_pytorch.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 preprocess_spnasnet_100_pytorch.py  /root/datasets/imagenet/val ./prep_bin
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 get_info.py bin ./prep_bin ./spnasnet_100_val.info 224 224
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
source /usr/local/Ascend/ascend-lastest/set_env.sh
```
2.执行离线推理
```
/benchmark.${arch} -model_type=vision -device_id=0 -batch_size=1 -om_path=spnasnet_100_bs1.om -input_text_path=./spnasnet_100_val.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_device{0}，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。

## 6 精度对比

-   **[310离线推理TopN精度](#61-310离线推理TopN精度)**
-   **[310P离线推理TopN精度](#62-310P离线推理TopN精度)** 
-   **[开源TopN精度](#63-开源TopN精度)**  
-   **[精度对比](#64-精度对比)**  

### 6.1 310离线推理TopN精度统计

后处理统计TopN精度

调用vision_metric_ImageNet.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
```
python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ /root/datasets/imagenet/val_label.txt ./ result_bs1.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  

310精度结果：
|         | Top1 accuracy | Top5 accuracy |
|---------|---------------|---------------|
| 310     | 74.2%         | 91.94%        |

### 6.2 310P离线推理TopN精度统计

同310，调用vision_metric_ImageNet.py脚本：
```
python3.7 vision_metric_ImageNet.py result/dumpOutput_device0/ /root/datasets/imagenet/val_label.txt ./ result_bs1.json
```

310P精度结果：
|         | Top1 accuracy | Top5 accuracy |
|---------|---------------|---------------|
| 310P    | 74.21%        | 91.95%        |

### 6.3 开源TopN精度

[timm官网精度](https://rwightman.github.io/pytorch-image-models/results/)

| model        | Top1 accuracy   | Top1 accuracy   |
| ------------ | --------------- | --------------- |
| spnasnet_100 | 74.084%         | 91.818%         |

### 6.4 精度对比

将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度高于开源代码仓，故精度达标。

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
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=spnasnet_100_bs1.om -input_text_path=./spnasnet_100_val.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```

310上各batch的吞吐率：

batch1 310单卡吞吐率：678.142 * 4=2712.568 fps

batch4 310单卡吞吐率：1311.7 * 4=2712.568 fps

batch8 310单卡吞吐率：1351.14 * 4=5404.56 fps

batch16 310单卡吞吐率：1418.95 * 4=5675.8 fps

batch32 310单卡吞吐率：1319.95 * 4=5279.8 fps

batch64 310单卡吞吐率：1262.87 * 4=5051.48 fps

 **性能优化：**  
>没有遇到性能不达标的问题，故不需要进行性能优化

### 7.2 310P性能数据

同310，使用benchmark工具在整个数据集上推理获得性能数据：
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=spnasnet_100_bs1.om -input_text_path=./spnasnet_100_val.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```

Interface throughputRate:1743.58，即是batch1 310P单卡吞吐率  

310P上各batch的吞吐率：

batch4 310P单卡吞吐率：1743.58 fps

batch4 310P单卡吞吐率：5244.27 fps

batch8 310P单卡吞吐率：7196.11 fps

batch16 310P单卡吞吐率：6367.69 fps

batch32 310P单卡吞吐率：5872.96 fps 

batch64 310P单卡吞吐率：4873.73 fps 


### 7.3 T4性能数据

在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2 

使用benchmark工具在整个数据集上推理获得性能数据：
```
trtexec --onnx=spnasnet_100.onnx --fp16 --shapes=actual_input_1:1x3x224x224 --workspace=5000 --threads
```

T4上各batch的吞吐率：

batch4 T1单卡吞吐率：1766.07 * 1 = 1766.07 qps

batch4 T4单卡吞吐率：477.875 * 4 =  1911.5 qps

batch8 T4单卡吞吐率：333.879 * 8 =  2671.032 qps

batch16 T4单卡吞吐率：271.744 * 16 =  4347.904 qps

batch32 T4单卡吞吐率：128.717 * 32 =  4118.944 qps

batch32 T4单卡吞吐率：35.9231 * 64 =  2299.0784 qps

### 7.4 性能对比

性能对比表格如下：

|           | 310      | 310P    | T4        | 310P/310    | 310P/T4     |
| --------- | -------- | ------- | --------- | ----------- | ----------- |
| bs1       | 2712.568 | 1743.58 | 1766.07   | 0.642778356 | 0.98726551  |
| bs4       | 5246.8   | 5244.27 | 1911.5    | 0.999517801 | 2.74353649  |
| bs8       | 5404.56  | 7196.11 | 2671.032  | 1.331488595 | 2.694130958 |
| bs16      | 5675.8   | 6367.69 | 4347.904  | 1.121901758 | 1.464542455 |
| bs32      | 5279.8   | 5872.96 | 4118.944  | 1.112345165 | 1.425841186 |
| bs64      | 5051.48  | 4873.73 | 2299.0784 | 0.964812293 | 2.119862463 |
| 最优batch | 5675.8   | 7196.11 | 4347.904  | 1.267858275 | 1.655075641 |

最优的310P性能达到了最优的310性能的1.268倍，达到最优的T4性能的1.655倍。

**性能优化：**
性能已达标，不需要再优化。