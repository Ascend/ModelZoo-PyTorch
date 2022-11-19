# FaceBoxes Onnx模型端到端推理指导
- [FaceBoxes Onnx模型端到端推理指导](#FaceBoxes-onnx模型端到端推理指导)
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
		- [6.1 离线推理mAP精度](#61-离线推理mAP精度)
		- [6.2 开源mAP精度](#62-开源mAP精度)
		- [6.3 精度对比](#63-精度对比)
	- [7 性能对比](#7-性能对比)
		- [7.1 npu性能数据](#71-npu性能数据)


## 1 模型概述

-   **[论文地址](#11-论文地址)**  

-   **[代码地址](#12-代码地址)**  

### 1.1 论文地址
[FaceBoxes论文](https://arxiv.org/abs/1708.05234.pdf)  

### 1.2 代码地址
[FaceBoxes代码](https://github.com/zisianw/FaceBoxes.PyTorch)  

## 2 环境说明

-   **[深度学习框架](#21-深度学习框架)**  

-   **[python第三方库](#22-python第三方库)**  

### 2.1 深度学习框架
```
CANN 5.1.RC1
python = 3.7.5
torch = 1.12.0
```

### 2.2 python第三方库

```
numpy = 1.21.5
decorator = 5.1.1
sympy = 1.10.1
opencv-python = 4.6.0.66
tqdm = 4.64.0
Cpython = 0.29.32
```
**说明：** 
>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

-   **[pth转onnx模型](#31-pth转onnx模型)**  

-   **[onnx转om模型](#32-onnx转om模型)**  

### 3.1 pth转onnx模型

1.FaceBoxes模型代码下载
```
git clone https://github.com/zisianw/FaceBoxes.PyTorch.git
(commit_id=9bc5811fe8c409a50c9f23c6a770674d609a2c3a)
cd FaceBoxes.PyTorch
```
2.预训练模型获取。
```
到以下链接下载预训练模型：
(https://drive.google.com/file/d/1tRVwOlu0QtjvADQ2H7vqrRwsWEmaqioI) 
```

3.下载仓库中的本模型文件到FaceBoxes.PyTorch目录下

 **说明：**  
>注意目前ATC支持的onnx算子版本为11

4.执行pth2onnx脚本，生成onnx模型文件
```
#将FaceBoxesProd.pth模型转为faceboxes-b0_bs1.onnx模型.
python3.7 faceboxes_pth2onnx.py  --trained_model FaceBoxesProd.pth --save_folder faceboxes-b0.onnx  
```

 **模型转换要点：**  
### 3.2 onnx转om模型

1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.1.RC1 开发辅助工具指南 (推理) 01，ii.	注意先通过https://netron.app/ 查看onnx的输出节点名称，对应的进行更改--out_nodes里的参数 ，注意--soc_version中${chip_name}可通过npu-smi info指令查看，例：--soc_version=Ascend310P3

```
atc --framework=5 --model=faceboxes-b0.onnx --output=faceboxes-b0_bs1 --input_format=NCHW --input_shape="image:1,3,1024,1024" --log=debug --soc_version=Ascend${chip_name} --out_nodes="Reshape_134:0;Softmax_141:0" --enable_small_channel=1 --insert_op_conf=aipp.config

atc --framework=5 --model=faceboxes-b0.onnx --output=faceboxes-b0_bs16 --input_format=NCHW --input_shape="image:16,3,1024,1024" --log=debug --soc_version=Ascend${chip_name} --out_nodes="Reshape_134:0;Softmax_141:0" --enable_small_channel=1 --insert_op_conf=aipp.config
```

## 4 数据集预处理

-   **[数据集获取](#41-数据集获取)**  

-   **[数据集预处理](#42-数据集预处理)**  

-   **[生成数据集信息文件](#43-生成数据集信息文件)**  

### 4.1 数据集获取
该模型使用[FDDB官网](http://vis-www.cs.umass.edu/fddb/)的2845张验证集进行测试，图片存放在/root/datasets/data/FDDB，标签存放在FDDB_Evaluation/ground_truth。

### 4.2 数据集预处理
1.预处理脚本faceboxes_pth_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件
```
python3.7 faceboxes_pth_preprocess.py --dataset /root/datasets/data/FDDB --save-folder prep/
```
### 4.3 生成数据集信息文件
1.生成数据集信息文件脚本get_info.py 

2.执行生成数据集信息脚本，生成数据集信息文件
```
python3.7 get_info.py bin ./prep ./faceboxes_prep_bin.info 1024 1024
```
第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息
## 5 离线推理

-   **[benchmark工具概述](#51-benchmark工具概述)**  

-   **[离线推理](#52-离线推理)**  

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.1.RC1 推理benchmark工具用户指南 01
### 5.2 离线推理
1.设置环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:${LD_LIBRARY_PATH}
```
2.执行离线推理
```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./faceboxes-b0_bs1.om -input_text_path=./faceboxes_prep_bin.info -input_width=1024 -input_height=1024 -output_binary=True -useDvpp=False
```
输出结果默认保存在当前目录result/dumpOutput_deviceX(X为对应的device_id)，每个输入对应一个_X.txt文件的输出。

## 6 精度对比

-   **[离线推理mAP精度](#61-离线推理mAP精度)**  
-   **[开源mAP精度](#62-开源mAP精度)**  
-   **[精度对比](#63-精度对比)**  

### 6.1 离线推理mAP精度

后处理统计mAP精度
```
dos2unix dlt_cuda.patch
git apply dlt_cuda.patch
./make.sh
```
调用faceboxes_pth_postprocess.py脚本,得到。
```
python3.7 faceboxes_pth_postprocess.py --save_folder FDDB_Evaluation/ --prep_info prep/ --prep_folder ./result/dumpOutput_device0/
```
第一个参数为结果文件所在的目录，第二个为前处理后的图片参数，第三个为om的输出目录。


调用evaluate.py可以获得AP精度数据。
```
cd FDDB_Evaluation
python3.7 setup.py install
python3.7 evaluate.py -g ./ground_truth
```

查看输出结果：result.txt
```
FDDB Dataset Average AP: 0.9484476599260253
```
经过对bs1的om测试，本模型batch16的精度没有差别，精度数据均如上。

### 6.2 开源mAP精度
[原代码仓公布精度](https://github.com/zisianw/FaceBoxes.PyTorch/blob/master/README.md)
```
Model                AP
FaceBoxesProd   0.9460
```
经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度于代码仓提供的pth测试的精度没有差别，精度数据均如上。

### 6.3 精度对比
将得到的om离线模型推理mAP精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  
 **精度调试：**  
>没有遇到精度不达标的问题，故不需要进行精度调试

## 7 性能对比

-   **[npu性能数据](#71-npu性能数据)**  

### 7.1 npu性能数据
1.benchmark工具在整个数据集上推理获得性能数据  

batch1性能：

batch1 310单卡吞吐率：186.048 * 4 = 744.192fps

batch16性能：

batch16 310单卡吞吐率：180.433 * 4 = 721.732fps


### 7.2 T4性能数据
在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2  
目前T4服务器安装的cuda,cudnn,TensorRT版本如上 
batch1性能：
```
trtexec --onnx=faceboxes-b0_bs1.onnx --fp16 --shapes=image:1x3x1024x1024 --threads
```
gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch。其中--fp16是算子精度，目前算子精度只测--fp16的。注意--shapes是onnx的输入节点名与shape，当onnx输入节点的batch为-1时，可以用同一个onnx文件测不同batch的性能，否则用固定batch的onnx测不同batch的性能不准  

batch1 t4单卡吞吐率：1000/(0.886072/1)=1128.576fps  

batch16性能：
```
trtexec --onnx=faceboxes-b0_bs1.onnx --fp16 --shapes=image:16x3x1024x1024 --threads
```
batch16 t4单卡吞吐率：1000/(8.31014/16)=1925.359fps  

### 7.3 性能对比

|      | 310 | 310P | T4 |
| -----|  ----| ---- | ---- |
| bs1 | 744 | 1943 | 1128.576 |
| bs4 | 727.2 | 1844 | 1719.003 |
| bs8 | 732 | 1949| 1843.038 |
| bs16 | 721.6 | 1841| 1925.359 |
| bs32 | 718 | 1819| 1955.679 |
| bs64 | 751.2 | 1767| 1981.614 |

已使用最新蓝区社区CANN包版本，且根据离线推理指导书未找到可以优化的点。