# PCB Onnx模型端到端推理指导
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
	-   [6.1 开源TopN精度](#62-开源TopN精度)
	-   [6.2 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
 

## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[PCB论文](https://arxiv.org/pdf/1711.09349.pdf)

分支为 : master

commit ID : e29cf54486427d1423277d4c793e39ac0eeff87c  

### 1.2 代码地址
[PCB开源仓代码](https://github.com/syfafterzy/PCB_RPP_for_reID)

## 2 环境说明
```
CANN:5.1.RC1
cuda:11.0
cudnn:8.2
TensoRT:7.2.3.4
```
-   **[深度学习框架](#21-深度学习框架)**  

### 2.1 深度学习框架
```
python==3.7.5
pytorch==1.8.1
torchvision==0.2.1
```

### 2.2 python第三方库

```
numpy == 1.21.6
scikit-learn == 0.24.1
opencv-python == 4.5.2.54
pillow == 8.2.0
onnx == 1.9.0
pillow == 8.2.0
skl2onnx == 1.8.0
h5py == 3.3.0 
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
[PCB预训练pth权重文件](https://ascend-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/face/PCB/PCB_3_7.pt)  
```
wget https://ascend-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/face/PCB/PCB_3_7.pt
```
 **说明：模型文件名为：PCB_3_7.pt  其md5sum值为：c5bc5ddabcbcc45f127ead797fe8cb35  PCB_3_7.pt**  
>获取的预训练模型放在本仓根目录下

2.编写pth2onnx脚本pth2onnx.py

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

3.执行pth2onnx脚本，生成onnx模型文件
```
python3.7 pth2onnx.py           #将PCB_3_7.pt模型转为PCB.onnx模型
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
atc --framework=5 --model=./models/PCB.onnx --output=PCB_bs1 --input_format=NCHW --input_shape="input_1:1,3,384,128" --log=debug --soc_version=Ascend${chip_name}
```
${chip_name}可通过`npu-smi info`查看，例如310P3
![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)
## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[Market数据集](https://pan.baidu.com/s/1ntIi2Op?_at_=1622802619466)的19732张验证集进行测试。数据集下载后，解压放到./datasets目录下。

### 4.2 数据集预处理
1.预处理脚本PCB_pth_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 PCB_pth_preprocess.py -d market -b 1 --height 384 --width 128 --data-dir ./datasets/Market-1501/ -j 4
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info_Ascend310.sh

2.执行生成数据集信息脚本，生成数据集信息文件
```
sh get_info_Ascend310.sh
```
在get_info_Ascend310.sh文件中调用华为提供的开源工具获取bin文件的路径和尺寸信息，该工具的第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)
### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理
```
sudo ./benchmark_tools/benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./PCB.om -input_text_path=./gallery_preproc_data_Ascend310.info -input_width=128 -input_height=384 -output_binary=True -useDvpp=False
sudo mv ./result/dumpOutput_device0 ./result/dumpOutput_device0_gallery
sudo mv ./result/perf_vision_batchsize_1_device_0.txt ./result/gallery_perf_vision_batchsize_1_device_0.txt
```
```
sudo ./benchmark_tools/benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./PCB.om -input_text_path=./query_preproc_data_Ascend310.info -input_width=128 -input_height=384 -output_binary=True -useDvpp=False
sudo mv ./result/dumpOutput_device0 ./result/dumpOutput_device0_query
sudo mv ./result/perf_vision_batchsize_1_device_0.txt ./result/query_perf_vision_batchsize_1_device_0.txt
```
输出结果默认保存在当前目录result/dumpOutput_device0下，由于需要通过om模型提取两组特征，因此根据输入图片类型（querry或gallery）分别重命名文件夹。
3.特征图后处理

```
python ./PCB_pth_postprocess.py -q ./result/dumpOutput_device0_query -g ./result/dumpOutput_device0_gallery -d market --data-dir ./datasets/Market-1501/
```
对om模型提取的特征做后处理并统计精度，结果如下：
```
{'title': 'Overall statistical evaluation', 'value': [{'key': 'Number of images', 'value': '15913'}, {'key': 'Number of classes', 'value': '751'}, {'key': 'Top-1 accuracy', 'value': '92.1%'}, {'key': 'Top-5 accuracy', 'value': '96.9%'}, {'key': 'Top-10 accuracy', 'value': '98.1%'}]}
```
## 6 精度对比

-   **[开源TopN精度](#61-开源TopN精度)**  
-   **[精度对比](#62-精度对比)**  

### 6.1 开源TopN精度
```
CMC Scores  market1501
  top-1          92.1%
  top-5          96.9%
  top-10         98.1%
```
### 6.2 精度对比
将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[310性能数据](#71-310性能数据)**  
-   **[310P性能数据](#72-310P性能数据)**  
-   **[T4性能数据](#73-T4性能数据)**  
-   **[性能对比](#74-性能对比)**  

### 7.1 310性能数据
benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准。  

1.benchmark工具在整个数据集上推理获得性能数据  

    ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=/home/zhouyc/original/PCB_bs1.om -input_text_path=./query_preproc_data_Ascend310.info -input_width=128 -input_height=384 -output_binary=True -useDvpp=False
batch1的性能，benchmark工具在整个数据集上推理后生成result/query_perf_vision_batchsize_1_device_0.txt.txt：
```
-----------------Performance Summary------------------
[e2e] throughputRate: 104.078, latency: 32360.4
[data read] throughputRate: 441.807, moduleLatency: 2.26343
[preprocess] throughputRate: 431.498, moduleLatency: 2.31751
[infer] throughputRate: 106.343, Interface throughputRate: 137.422 moduleLatency: 8.68511
[post] throughputRate: 106.341, moduleLatency: 9.40375
```
Interface throughputRate: 137.422，137.422*4=549.688即是batch1 310单卡吞吐率

2.310P上各batch的吞吐率：
```
batch1 310单卡吞吐率：549.688 fps

batch4 310单卡吞吐率：1296.72 fps

batch8 310单卡吞吐率：1375.24 fps

batch16 310单卡吞吐率：1411.96 fps

batch32 310单卡吞吐率：1212.25 fps

batch64 310单卡吞吐率：1188.12 fps
```

### 7.2 310P性能数据
同310，使用benchmark工具在整个数据集上推理获得性能数据：

310P上各batch的吞吐率：
```
batch1 310P单卡吞吐率：352.617 fps

batch4 310P单卡吞吐率：1512.1 fps

batch8 310P单卡吞吐率：1897.91 fps

batch16 310P单卡吞吐率：1492.53 fps

batch32 310P单卡吞吐率：1706.04 fps

batch64 310P单卡吞吐率：1857.43 fps
```
310P_aoe上各batch的吞吐率：
```
batch1 310P单卡吞吐率：1033.16 fps

batch4 310P单卡吞吐率：2364.58 fps

batch8 310P单卡吞吐率：2228.24 fps

batch16 310P单卡吞吐率：1999.29 fps

batch32 310P单卡吞吐率：2002.7 fps

batch64 310P单卡吞吐率：2039.35 fps
```
### 7.3 T4性能数据
在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2

使用benchmark工具在整个数据集上推理获得性能数据：

    trtexec --onnx=./models/PCB.onnx --shapes=input_1:64x3x384x128 --threads --fp16

gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch
```
[06/18/2022-21:13:40] [I] GPU Compute
[06/18/2022-21:13:40] [I] min: 37.9453 ms
[06/18/2022-21:13:40] [I] max: 46.219 ms
[06/18/2022-21:13:40] [I] mean: 40.5553 ms
[06/18/2022-21:13:40] [I] median: 40.0248 ms
[06/18/2022-21:13:40] [I] percentile: 46.219 ms at 99%
[06/18/2022-21:13:40] [I] total compute time: 3.822 s
```
batch64 t4单卡吞吐率：1000x1/(40.5553 /64)= 1578.09fps

T4上各batch的吞吐率：
```
batch4 T1单卡吞吐率：749.002 fps

batch4 T4单卡吞吐率：1209.63 fps

batch8 T4单卡吞吐率：1383.09 fps

batch16 T4单卡吞吐率：1495.91 fps

batch32 T4单卡吞吐率：1554.99 fps

batch32 T4单卡吞吐率：1578.09 fps
```

### 7.4 性能对比
|     |  310  | 310P  | 310P_aoe |T4  |310P_aoe/310  |310P_aoe/T4  |
|  ----  |  ----  | ----  | ----  |----  |----  |----  |
|bs1| 549.688  | 352.617 |1033.16  |749.002  |1.879538938  |1.37938224  |
|bs4|1296.72|	1512.1|	2364.58|	1209.63|	1.82350292	|	1.95479237|
|bs8|1375.24|	1897.91|	2228.24|	1383.09	|1.620260086|	1.61106487|
|bs16|1411.96	|1492.53|	1999.29	|1495.91|	1.415967874	|1.33650037|
|bs32|1212.25|	1706.04|	2002.7|	1554.99	|1.652054695|	1.28791759|
|bs64|1188.12	|1857.43	|2039.35|	1578.09	|1.71644542	|1.29228829|
| | | | | | |
|最优bs|	1411.96|	1897.91|	2364.58	|1578.09	|1.67468|	1.49838|