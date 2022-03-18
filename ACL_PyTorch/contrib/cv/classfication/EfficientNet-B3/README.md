# EfficientNet-B3 模型端到端推理指导
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
[Efficientnet论文](https://arxiv.org/abs/1905.11946)  

### 1.2 代码地址
[Efficientnet代码](https://github.com/facebookresearch/pycls)  
branch:master  
commit id:f20820e01eef7b9a47b77f13464e3e77c44d5e1f

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
[EfficientNet-B3预训练pth权重文件](https://dl.fbaipublicfiles.com/pycls/dds_baselines/161305060/EN-B3_dds_8gpu.pyth)  
文件md5sum: 4c809d9cb292ce541f278d11899e7b38 
```
wget https://dl.fbaipublicfiles.com/pycls/dds_baselines/161305060/EN-B3_dds_8gpu.pyth
```

2.下载efficientnet源码：  
```
git clone https://github.com/facebookresearch/pycls
cd pycls  
git reset f20820e01eef7b9a47b77f13464e3e77c44d5e1f --hard  
cd ..
```
如果使用补丁文件修改了模型代码则将补丁打入模型代码，如果需要引用模型代码仓的类或函数通过sys.path.append(r"./pycls")添加搜索路径。

3.编写pth2onnx脚本efficientnetB3_pth2onnx.py  
 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 efficientnetB3_pth2onnx.py EN-B3_dds_8gpu.pyth ./pycls/configs/dds_baselines/effnet/EN-B3_dds_8gpu.yaml efficientnetB3.onnx
```

### 3.2 onnx转om模型

1.设置环境变量
```
source env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01
```
atc --framework=5 --model=./efficientnetB3.onnx --input_format=NCHW --input_shape="image:16,3,300,300" --output=efficientnetB3_bs16 --log=debug --soc_version=Ascend310 --auto_tune_mode="RL,GA" 
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
1.预处理脚本imagenet_torch_preprocess.py


2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 imagenet_torch_preprocess.py efficientnetB3 /root/datasets/imagenet/val ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本gen_dataset_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 gen_dataset_info.py bin ./prep_dataset ./efficientnetb3_prep_bin.info 300 300
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
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=efficientnetB3_bs16.om -input_text_path=./efficientnetb3_prep_bin.info -input_width=300 -input_height=300 -output_binary=False -useDvpp=False
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
python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ /root/datasets/imagenet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：
```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "77.57%"}, {"key": "Top2 accuracy", "value": "87.3%"}, {"key": "Top3 accuracy", "value": "90.57%"}, {"key": "Top4 accuracy", "value": "92.34%"}, {"key": "Top5 accuracy", "value": "93.46%"}]}
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上。

### 6.2 开源TopN精度
[pycls实现精度](https://github.com/facebookresearch/pycls/blob/master/dev/model_error.json)  
```
Model               Acc@1     Acc@5
EfficientNet-B3     77.53     93.486  
```
### 6.3 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降不超过1%，故精度达标。

 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试。


## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**   

### 7.1 npu性能数据
 
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。   
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt： 
```
[e2e] throughputRate: 88.8383, latency: 562820
[data read] throughputRate: 91.6623, moduleLatency: 10.9096
[preprocess] throughputRate: 91.5715, moduleLatency: 10.9204
[infer] throughputRate: 89.0488, Interface throughputRate: 120.557, moduleLatency: 10.9898
[post] throughputRate: 89.0486, moduleLatency: 11.2298
```
Interface throughputRate: 120.557，120.557x4=482.228即是batch1 310单卡吞吐率。  
batch16的性能：  
benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：
```
[e2e] throughputRate: 115.546, latency: 432730
[data read] throughputRate: 119.201, moduleLatency: 8.38921
[preprocess] throughputRate: 119.109, moduleLatency: 8.39568
[infer] throughputRate: 115.783, Interface throughputRate: 168.121, moduleLatency: 8.42258
[post] throughputRate: 7.23632, moduleLatency: 138.192
```
Interface throughputRate: 168.121，168.121x4=672.484即是batch16 310单卡吞吐率
  
batch4性能：
```
./benchmark.x86_64 -round=20 -om_path=efficientnetB3_bs4.om -device_id=3 -batch_size=4
```
```
[INFO] Dataset number: 19 finished cost 25.448ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_efficientnetB3_bs4_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 157.113samples/s, ave_latency: 6.41437ms
----------------------------------------------------------------
```
batch4 310单卡吞吐率：157.113x4=628.452fps  
batch8 性能：
```
./benchmark.x86_64 -round=20 -om_path=efficientnetB3_bs8.om -device_id=3 -batch_size=8
```
```
[INFO] Dataset number: 19 finished cost 48.451ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_efficientnetB3_bs8_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 165.283samples/s, ave_latency: 6.09332ms
----------------------------------------------------------------
```
batch8 310单卡吞吐率：165.283x4=661.132fps    
batch32性能：
```
./benchmark.x86_64 -round=20 -om_path=efficientnetB3_bs32.om -device_id=3 -batch_size=32
```
```
[INFO] Dataset number: 19 finished cost 186.99ms
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_efficientnetB3_bs32_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 171.244samples/s, ave_latency: 5.85112ms
----------------------------------------------------------------
```
batch32 310单卡吞吐率：171.244x4=684.976fps  

 **性能优化：**  
> 从profiling数据的op_statistic_0_1.csv看出影响性能的是Conv2D，ReduceMeanD，Mul算子，从op_summary_0_1.csv可以看出单个算子aicore耗时都不高，故使用autotune优化  
对于bs16，不使用autotune的单卡吞吐率为583.4fps，使用autotune后单卡吞吐率为672.484fps。使用autotune后bs16性能达标

