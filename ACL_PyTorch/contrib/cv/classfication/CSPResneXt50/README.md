# CSPResNeXt50 Onnx模型端到端推理指导
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
[CSPResNeXt50论文](https://arxiv.org/pdf/1911.11929v1.pdf)  

### 1.2 代码地址
[CSPResNeXt50代码](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/cspnet.py)  
branch:master  
commit id:d584e7f617a4d0f1a0b4838227bd1f8852dfa236  


## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.0.2.alpha003

pytorch = 1.5.0
torchvision = 0.6.0
onnx = 1.11.1
```
### 2.2 python第三方库

```
numpy == 1.22.3
Pillow == 8.3.2
opencv-python == 4.5.3.56
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
[CSPResNeXt50预训练pth权重文件](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/cspresnext50_ra_224-648b4713.pth)  
文件md5sum: 9a123c2b8a4eafd42926d2f3c105ed3d  

```
wget http://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/cspresnext50_ra_224-648b4713.pth
```
2.cspresnext50模型代码在代码仓中

```
git clone https://github.com/rwightman/pytorch-image-models.git 
```
 3.编写pth2onnx脚本cspresnext50_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件

```
python3.7 cspresnext_pth2onnx.py cspresnext50_ra_224-648b4713.pth cspresnext.onnx
```

 **模型转换要点：**  
>此模型转换为onnx不需要修改开源代码仓代码，故不需要特殊说明  

### 3.2 onnx转om模型

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01
```
atc --framework=5 --model=./cspresnext.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=cspresnext_bs1 --log=debug --soc_version=Ascend310
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
图片与标签分别存放在/opt/npu/imagenet/val与/opt/npu/imageNet/val_label.txt。

### 4.2 数据集预处理
1.预处理脚本imagenet_torch_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 cspresnext_torch_preprocess.py /opt/npu/imageNet/val ./prep_dataset
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 get_info.py bin ./prep_dataset ./cspresnext_prep_bin.info 224 224
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.2 推理benchmark工具用户指南 01
### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=cspresnext_bs1.om -input_text_path=./cspresnext_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
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
python3.7 imagenet_acc_eval.py result/dumpOutput_device0/ /opt/npu/imageNet/val_label.txt ./ result.json
```
第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。  
查看输出结果：
```
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "80.06%"}, {"key": "Top2 accuracy", "value": "89.18%"}, {"key": "Top3 accuracy", "value": "92.33%"}, {"key": "Top4 accuracy", "value": "93.98%"}, {"key": "Top5 accuracy", "value": "94.94%"}]}
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上   

### 6.2 开源精度
[论文精度](https://arxiv.org/pdf/1911.11929v1.pdf)

```
Model               	Acc@1     Acc@5
CSPResNeXt-50		     77.9      94.0
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
[e2e] throughputRate: 203.598, latency: 245582
[data read] throughputRate: 216.393, moduleLatency: 4.62122
[preprocess] throughputRate: 216.18, moduleLatency: 4.62577
[infer] throughputRate: 204.451, Interface throughputRate: 304.088, moduleLatency: 4.24378
[post] throughputRate: 204.451, moduleLatency: 4.89115
```
Interface throughputRate: 304.088，304.088x4既是batch1 310单卡吞吐率  
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：  

```
[e2e] throughputRate: 130.345, latency: 383598
[data read] throughputRate: 130.66, moduleLatency: 7.65348
[preprocess] throughputRate: 130.564, moduleLatency: 7.65911
[infer] throughputRate: 130.561, Interface throughputRate: 421.719, moduleLatency: 3.85317
[post] throughputRate: 8.15993, moduleLatency: 122.55
```
Interface throughputRate: 421.719，421.719x4既是batch16 310单卡吞吐率  
batch4性能：  
 ./benchmark.x86_64 -round=20 -batch_size=4 -device_id=0 -om_path=cspresnext_bs4.om
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_cspresnext_bs4_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 440.606samples/s, ave_latency: 2.304ms
----------------------------------------------------------------

```
batch4 310单卡吞吐率：440.606x4=1762.424 fps  
batch8性能：
```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_cspresnext_bs8_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 464.07samples/s, ave_latency: 2.16864ms
----------------------------------------------------------------

```
batch8 310单卡吞吐率：464.07x4=1856.28fps  
batch32性能：

```
[INFO] PureInfer result saved in ./result/PureInfer_perf_of_cspresnext_bs32_in_device_0.txt
-----------------PureInfer Performance Summary------------------
[INFO] ave_throughputRate: 369.97samples/s, ave_latency: 2.7061ms
----------------------------------------------------------------
```
batch32 310单卡吞吐率：369.97x4=1479.88fps  
  
 **性能优化：**  

1.从profiling数据的op_statistic_0_1.csv看出影响性能的是TransData，StridedSliceD，Conv2D算子，Conv2D算子不存在问题，由于格式转换om模型StridedSliceD前后需要有TransData算子，从op_summary_0_1.csv可以看出单个TransData或Transpose算子aicore耗时，确定是否可以优化  
2.蓝区社区版本CANN 5.0.2.alpha003优化了StridedSliceD，可以使bs1,4,8,16达标  
3.修改five_2_four.py与four_2_five.py优化TransData可以使bs32达标   
> five_2_four.py:9928  
> 修改如下：  
elif dst_format.lower() == "nchw" and dst_shape in [[2560, 512, 4, 26], [2560, 512, 1, 26], [2560, 256, 8, 25],  
                                                    [16, 240, 7, 7], [16, 120, 14, 14],  
                                                    [32,2048,7,7],[32,1024,14,14],[32,256,56,56],[32,512,28,28]]:
														
> four_2_five.py:1219  
修改如下：  
if src_format.upper() == "NCHW" and shape_input in [[16, 240, 7, 7], [16, 120, 14, 14],  
                                                    [32,1024,7,7],[32,3,224,224],[32,128,56,56]] and dtype_input == "float16":  
>

优化后测得单卡吞吐率bs32: 369.97x4=1479.88fps  
同理优化bs1 4 8 16的Transdata，上面记录的都是优化后的性能数据