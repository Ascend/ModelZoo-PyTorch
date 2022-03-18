# ReID Onnx模型端到端推理指导
- [ReID Onnx模型端到端推理指导](#ReID-onnx模型端到端推理指导)
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
		- [6.1 离线推理精度](#61-离线推理精度)
		- [6.2 开源精度](#62-开源精度)
		- [6.3 精度对比](#63-精度对比)
	- [7 性能对比](#7-性能对比)
		- [7.1 npu性能数据](#71-npu性能数据)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[ReID论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1903.07071.pdf)  

### 1.2 代码地址
[ReID代码](https://github.com/michuanhaohao/reid-strong-baseline)  
branch:master  
commit_id: 3da7e6f03164a92e696cb6da059b1cd771b0346d

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN == 5.0.1
torch == 1.5.0
torchvision == 0.6.0
onnx == 1.7.0

```

### 2.2 python第三方库

```
numpy == 1.20.3
Pillow == 8.2.0
opencv-python == 4.5.2.54
yacs == 0.1.8
pytorch-ignite == 0.4.5

```


## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.获取，修改与安装ReID模型代码  

```
git clone https://github.com/michuanhaohao/reid-strong-baseline
cd reid-strong-baseline  
如果修改了模型代码   
patch -p1 < ../{model_name}.diff  
pth2onnx等脚本需要引用模型代码的类或函数，可通过sys.path.append(r"./reid-strong-baseline")添加搜索路径的方式   
cd ..  
```

2.下载.pth权重文件

[pth权重文件](https://drive.google.com/open?id=1hn0sXLZ5yJcxtmuY-ItQfYD7hBtHwt7A)

[网盘pth权重文件，提取码：v5uh](https://pan.baidu.com/s/1ohWunZOrOGMq8T7on85-5w)  

文件名：market_resnet50_model_120_rank1_945.pth  
md5sum：0811054928b8aa70b6ea64e71ef99aaf


3.编写pth2onnx脚本ReID_pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行ReID_pth2onnx.py脚本，生成onnx模型文件
```
python3.7 ReID_pth2onnx.py --config_file='reid-strong-baseline/configs/softmax_triplet_with_center.yml' MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('market_resnet50_model_120_rank1_945.pth')" TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('no')"  
```

 **模型转换要点：**  
> 加上TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('no')"导出的onnx可以推理测试性能  
> 不加上TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('no')"导出的onnx转换的om精度与官网精度一致

### 3.2 onnx转om模型

1.设置环境变量
```
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/

```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01
```
atc --framework=5 --model=ReID.onnx --output=ReID_bs1 --input_format=NCHW --input_shape="image:1,3,256,128" --log=debug --soc_version=Ascend310
atc --framework=5 --model=ReID.onnx --output=ReID_bs16 --input_format=NCHW --input_shape="image:16,3,256,128" --log=debug --soc_version=Ascend310
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
[Market1501数据集](http://www.liangzheng.org/Project/project_reid.html)

### 4.2 数据集预处理

执行两次预处理脚本ReID_preprocess.py，分别生成数据集query和数据集gallery预处理后的bin文件
```
python3.7 ReID_preprocess.py /root/datasets/market1501/query prep_dataset_query
python3.7 ReID_preprocess.py /root/datasets/market1501/bounding_box_test prep_dataset_gallery
mv prep_dataset_gallery/* prep_dataset_query/
```
### 4.3 生成数据集信息文件
执行gen_dataset_info.py脚本，生成数据集信息文件
```
python3.7 gen_dataset_info.py bin prep_dataset_query prep_bin.info 128 256
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
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/

```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./ReID_bs1.om -input_text_path=./prep_bin.info -input_width=128 -input_height=256 -output_binary=True -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_deviceX(X为对应的device_id)，每个输入对应一个_X.bin文件的输出，shape为bs*2048，数据类型为FP32.

## 6 精度对比

-   **[离线推理IoU精度](#61-离线推理IoU精度)**  
-   **[开源IoU精度](#62-开源IoU精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理精度

后处理统计精度，后处理脚本ReID_postprocess.py

```
python3.7 ReID_postprocess.py --query_dir=/root/datasets/market1501/query --gallery_dir=/root/datasets/market1501/bounding_box_test --pred_dir=./result/dumpOutput_device0
```
第一个为query数据集输入，第二个为gallery数据集输入，第三个为benchmark推理输出结果目录

查看输出结果：
```
RANK-1: 0.945
mAP: 0.859
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上。

### 6.2 开源精度
[原代码仓公布精度](https://github.com/michuanhaohao/reid-strong-baseline/blob/master/README.md)
```
Model	RANK-1   mAP
ReID	0.945     0.859 
```
### 6.3 精度对比
将得到的om离线模型推理IoU精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。  
1.benchmark工具在整个数据集上推理获得性能数据  
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：  
```
[e2e] throughputRate: 141.4, latency: 139547
[data read] throughputRate: 142.595, moduleLatency: 7.01285
[preprocess] throughputRate: 142.32, moduleLatency: 7.0264
[infer] throughputRate: 142.234, Interface throughputRate: 361.547, moduleLatency: 3.2183
[post] throughputRate: 142.233, moduleLatency: 7.03069

```
Interface throughputRate: 361.547，361.547x4=1446.188既是batch1 310单卡吞吐率    
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_0.txt：  
```
[e2e] throughputRate: 131.964, latency: 25522
[data read] throughputRate: 140.716, moduleLatency: 7.10649
[preprocess] throughputRate: 138.915, moduleLatency: 7.19867
[infer] throughputRate: 138.589, Interface throughputRate: 647.14, moduleLatency: 2.49923
[post] throughputRate: 8.68207, moduleLatency: 115.18
```
Interface throughputRate: 647.14，647.14x4=2588.56既是优化后batch16 310单卡吞吐率   
batch4性能：
```
[e2e] throughputRate: 130.2, latency: 151552
[data read] throughputRate: 131.306, moduleLatency: 7.61581
[preprocess] throughputRate: 131.041, moduleLatency: 7.63118
[infer] throughputRate: 130.947, Interface throughputRate: 469.146, moduleLatency: 3.10762
[post] throughputRate: 32.7368, moduleLatency: 30.5467

```
batch4 310单卡吞吐率：469.146x4=1984.584  
batch8性能：
```
[e2e] throughputRate: 143.491, latency: 137514
[data read] throughputRate: 145.468, moduleLatency: 6.87437
[preprocess] throughputRate: 145.165, moduleLatency: 6.8887
[infer] throughputRate: 145.023, Interface throughputRate: 551.773, moduleLatency: 2.77813
[post] throughputRate: 18.1315, moduleLatency: 55.1527

```
batch8 310单卡吞吐率：551.773x4=2206.932  
batch32性能：
```
[e2e] throughputRate: 283.934, latency: 69495
[data read] throughputRate: 358.044, moduleLatency: 2.79295
[preprocess] throughputRate: 356.976, moduleLatency: 2.80131
[infer] throughputRate: 288.848, Interface throughputRate: 517.032, moduleLatency: 2.89759
[post] throughputRate: 9.03173, moduleLatency: 110.721

```
batch32 310单卡吞吐率：517.032x4=2068.128  

#### 性能优化

>从profiling数据的op_statistic_0_1.csv看出影响性能的是Conv2D，Cast，BatchNorm算子，从op_summary_0_1.csv可以看出单个算子aicore耗时都不高，故使用autotune优化
对于bs16，不使用autotune的单卡吞吐率为2026.18fps，使用autotune后单卡吞吐率为2588.56fps。使用autotune后bs16性能达标
