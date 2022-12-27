# Efficient-3DCNNs模型PyTorch离线推理指导

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
	-   [4.3 生成数据集二进制文件和信息文件](#43-生成数据集二进制文件和信息文件)
-   [5 离线推理](#5-离线推理)
	-   [5.1 获取ais_infer推理工具](#51-获取ais_infer推理工具)
	-   [5.2 离线推理](#52-离线推理)
-   [6 精度对比](#6-精度对比)
	-   [6.1 离线推理topk精度统计](#61-离线推理topk精度统计)
	-   [6.2 开源topk精度](#62-开源topk精度)
	-   [6.3 精度对比](#63-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 npu性能数据](#71-npu性能数据)



## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[Efficient-3DCNNs论文](https://arxiv.org/pdf/1904.02422.pdf)  

### 1.2 代码地址
[Efficient-3DCNNs代码](https://github.com/okankop/Efficient-3DCNNs)  
branch:master  
commit_id:d60c6c48cf2e81380d0a513e22e9d7f8467731af

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC2
torch==1.5.0
torchvision==0.6.0
onnx==1.7.0
python==3.7.5

```

### 2.2 python第三方库

```
numpy==1.18.5
Pillow==7.2.0
opencv-python==4.5.3.56
```

**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.准备pth权重文件  
使用训练好的pth权重文件

[ucf101_mobilenetv2_1.0x_RGB_16_best.pth](https://drive.google.com/drive/folders/1u4DO7kjAQP6Zdh8CN65iT5ozp11mvE-H?usp=sharing)  

2.导出onnx文件
安装开源模型代码
```
 git clone https://github.com/okankop/Efficient-3DCNNs
```
导出onnx文件
```
python3.7 Efficient-3DCNNs_pth2onnx.py ucf101_mobilenetv2_1.0x_RGB_16_best.pth Efficient-3DCNNs.onnx
```
获得Efficient-3DCNNs.onnx文件

3.模型简化
```
python3.7 -m onnxsim Efficient-3DCNNs.onnx Efficient-3DCNNs_sim.onnx --input-shape "16,3,16,112,112" --dynamic-input-shape  
```
得到简化后的Efficient-3DCNNs_sim.onnx模型

 **说明：**  
>注意目前ATC支持的onnx算子版本为11


### 3.2 onnx转om模型

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/Ascend/driver/lib64/driver/
```

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 
${chip_name}可通过npu-smi info指令查看，例：310P3

![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

```
atc \
--model=Efficient-3DCNNs_sim.onnx \
--framework=5 \
--output=Efficient-3DCNNs_bs1 \
--input_format=NCDHW \
--input_shape="image:1,3,16,112,112" \
--log=error \
--soc_version=${chip_name} \
```
运行成功后生成“Efficient-3DCNNs_bs1.om”模型文件。

如需生成其他batchsize的om模型，请修改参数input_shape中的第0维，并修改参数output，例如生成batchsize=4的om模型，命令如下：

```
atc \
--model=Efficient-3DCNNs_sim.onnx \
--framework=5 \
--output=Efficient-3DCNNs_bs4 \
--input_format=NCDHW \
--input_shape="image:4,3,16,112,112" \
--log=error \
--soc_version=${chip_name} \
```
## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集二进制文件和信息文件](#43-生成数据集二进制文件和信息文件)**  

### 4.1 数据集获取
本模型基于UCF-101训练和推理，UCF-101是一个轻量的动作识别数据汇集，包含101种动作的短视频。

[获取UCF-101数据集](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)

下载后，上传数据集到服务器任意目录并解压（如：/opt/npu/）。

### 4.2 数据集预处理
在任意目录下载并解压ffmpeg（如：/home/HwHiAiUser/datasets/）
```
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz
xz -d ffmpeg-git-amd64-static.tar.xz
tar -xvf ffmpeg-git-amd64-static.tar
```

将ffmpeg文件添加到/bin目录下
```
cd /bin 
ln -s /home/HwHiAiUser/datasets/ffmpeg-git-20210908-amd64-static/ffmpeg ffmpeg 
ln -s /home/HwHiAiUser/datasets/ffmpeg-git-20210908-amd64-static/ffprobe ffprobe
```
其中ffmpeg-git-20210908-amd64-static文件夹名称可能根据下载版本不同有差异，需自行替换为所下载版本的名称

在/home/HwHiAiUser/datasets中新建rawframes文件夹，后运行Efficient-3DCNNs文件夹下的脚本将数据格式转为从视频帧中提取的图片。
```
 mkdir datasets
 cd datasets
 mkdir rawframes
 cd /home/HwHiAiUser/Efficient-3DCNNs
 python3.7 Efficient-3DCNNs/utils/video_jpg_ucf101_hmdb51.py /opt/npu/UCF-101/ /home/HwHiAiUser/datasets/rawframes
 python3.7 Efficient-3DCNNs/utils/n_frames_ucf101_hmdb51.py /home/HwHiAiUser/datasets/rawframes
```

### 4.3 生成数据集二进制文件和info文件
数据预处理将图片数据转换为模型输入的二进制数据。

将原始数据（.jpg）转化为二进制文件（.bin）。执行Efficient-3DCNNs_preprocess.py脚本。

```
python3.7 Efficient-3DCNNs_preprocess.py --video_path=/home/HwHiAiUser/datasets/rawframes --annotation_path=./Efficient-3DCNNs/annotation_UCF101/ucf101_01.json --output_path=bin1 --info_path=ucf101_bs1.info --inference_batch_size=1
```

参数说明：

--video_path：jpg数据最上层目录。

--annotation_path：数据集信息路径。

--output_path：输出二进制目录。

--info_path：输出info文件。

--inference_batch_size：推理batch_size。

运行完预处理脚本会在当前目录输出ucf101_bs1.info文件和bin1二进制文件夹，用于推理

## 5 离线推理

-   **[获取ais_bench推理工具](#51-获取ais_bench推理工具)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 获取ais_bench推理工具

请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

### 5.2 离线推理
昇腾芯片上执行，执行时使npu-smi info查看设备状态，确保device空闲

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.执行离线推理

在当前目录下新建推理结果文件夹

```
mkdir result
```
执行推理
```
python3 -m ais_bench --model ./Efficient-3DCNNs/Efficient-3DCNNs_bs1.om  --input ./Efficient-3DCNNs/bin1/ --output result --output_dirname out1
```
参数说明：

--model：需要进行推理的om模型。

--input：模型需要的输入，支持bin文件和目录。

--output：推理结果输出路径。默认会建立日期+时间的子文件夹保存输出结果 如果指定output_dirname 将保存到output_dirname的子文件夹下。

--output_dirname：推理结果输出子文件夹。可选参数。与参数output搭配使用，单独使用无效。设置该值时输出结果将保存到 output/output_dirname文件夹中。

## 6 精度对比

-   **[离线推理topk精度](#61-离线推理topk精度)**  
-   **[开源topk精度](#62-开源topk精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理topk精度统计

后处理统计topk精度

调用Efficient-3DCNNs_postprocess.py脚本与数据集标签ucf101_01.json比对，可以获得topk数据，结果保存在result.json中。
```
python3.7 Efficient-3DCNNs_postprocess.py --info_path=ucf101_bs1.info --result_path=/home/gc/result/out1  --annotation_path=Efficient-3DCNNs/annotation_UCF101/ucf101_01.json --acc_file=result.json
```
参数说明:

--result_path：推理结果的路径

--info_path：为前处理生成的info文件  
    
--ucf101_01.json：为标签数据
    
--result.json：为生成结果文件


### 6.2 开源topk精度

```
Model                   top1        top5   
Efficient-3DCNNs        0.81126     0.96299 
```
### 6.3 精度对比
将得到的om离线模型推理topk精度与该模型精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
1.ais_bench工具在整个数据集上推理获得性能数据  

| Throughput | 310 | 310P      | T4 | 310P/310 | 310P/T4 |     
|------------|-----|-----------|----|----------|---------|
| 1          | 256.4102   | 1001.4662 |  73.237  | 3.9057   | 13.6742 |
| 4          | 734.9563   | 1303.4864 | 212.0519   | 1.7735   | 6.1470  |
| 8          | 661.4303  | 1145.052  | 238.6378   | 1.7311   | 4.7982  |
| 16         | 760.0047  | 1255.7488 |  393.2701  | 1.6522   | 3.1930  |
| 32         | 770.1101  | 1299.9648 |  402.7771  | 1.6880   | 3.2275  |
| 64         | 770.1332  | 1290.3424 |  406.6693  | 1.6754   | 3.1729  |
|            |     |           |    |          |         |
| 最优batch    |  770.1332   | 1303.4864 |  406.6693  | 1.6925   | 3.2052  |

Interface throughputRate:1000 * batchsize/npu_compute_time.mean=1001.4662既是batch1 310P单卡吞吐率

**性能优化：**  

>没有遇到性能不达标的问题，故不需要进行性能优化
