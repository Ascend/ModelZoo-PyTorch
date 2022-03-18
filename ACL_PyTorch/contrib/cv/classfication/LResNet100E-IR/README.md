# LResNet100E-IR Onnx模型端到端推理指导

+ [1模型概述](#1 模型概述)

  + [1.1 论文地址](##1.1 论文地址)
  + [1.2 代码地址](##1.2 代码地址)

+ [2 环境说明](#2 环境说明)

  + [2.1 深度学习框架](##2.1 深度学习框架)
  + [2.2 python第三方库](##2.2 python第三方库)

+ [3 模型转换](#3 模型转换)

  + [3.1 pth转onnx模型](##3.1 pth转onnx模型)
  + [3.2 onnx转om模型](##3.2 onnx转om模型)

+ [4 数据集预处理](#4 数据集预处理)

  + [4.1 数据集获取](##4.1 数据集获取)
  + [4.2 数据集预处理](##4.2 数据集预处理)
  + [4.3 生成预处理数据集信息文件](##4.3 生成预处理数据集信息文件)

+ [5 离线推理](#5 离线推理)

  + [5.1 benchmark工具概述](##5.1 benchmark工具概述)
  + [5.2 离线推理](##5.2 离线推理)

+ [6 精度对比](#6 精度对比)

  + [6.1 离线推理精度统计](##6.1 离线推理精度统计)
  + [6.2 开源精度](##6.2 开源精度)
  + [6.3 精度对比](##6.3 精度对比)

+ [7 性能对比](#7 性能对比)

  + [7.1 npu性能数据](##7.1 npu性能数据)
  + [7.2 gpu和npu性能对比](##7.2 gpu和npu性能对比)

  

## 1 模型概述

### 1.1 论文地址

[LResNet100E-IR论文](https://arxiv.org/pdf/1801.07698.pdf )

### 1.2 代码地址

[LResNet100E-IR代码](https://github.com/TreB1eN/InsightFace_Pytorch )



## 2 环境说明

### 2.1 深度学习框架

```
torch==1.5.0
torchvision==0.6.0
onnx==1.10.1
onnx-simplifier==0.3.6
```

### 2.2 python第三方库

```
numpy==1.21.2
opencv-python==4.5.3.56
pillow==8.3.2
tqdm==4.62.2
scikit-learn==0.24.2
```



## 3 模型转换

### 3.1 pth转onnx模型

1.LResNet100E-IR模型代码下载

```bash
# 切换到工作目录
cd LResNet100E-IR

git clone https://github.com/TreB1eN/InsightFace_Pytorch.git ./LResNet
cd LResNet
patch -p1 < ../LResNet.patch
rm -rf ./work_space/* 
mkdir ./work_space/history && mkdir ./work_space/log && mkdir ./work_space/models && mkdir ./work_space/save
cd ..
```

2.获取模型权重，并放在工作目录的model文件夹下

OBS： [model_ir_se100.pth](obs://l-resnet100e-ir/infer/model_ir_se100.pth)  云盘：[model_ir_se100.pth](https://drive.google.com/file/d/1rbStth01wP20qFpot06Cy6tiIXEEL8ju/view?usp=sharing)

```bash
mkdir model
mv model_ir_se100.pth ./model/
```



3.使用 LResNet_pth2onnx.py 脚本将pth模型文件转为onnx模型文件

+ 参数1：pth模型权重的路径

+ 参数2：onnx模型权重的存储路径

+ 参数3：batch size

```bash.
python LResNet_pth2onnx.py ./model/model_ir_se100.pth ./model/model_ir_se100_bs1.onnx 1
python LResNet_pth2onnx.py ./model/model_ir_se100.pth ./model/model_ir_se100_bs16.onnx 16
```

4.使用 onnxsim 工具优化onnx模型

+ 参数1：输入的shape
+ 参数2：onnx模型权重的存储路径
+ 参数3：优化后onnx模型权重的存储路径

```
python -m onnxsim --input-shape="1,3,112,112" ./model/model_ir_se100_bs1.onnx ./model/model_ir_se100_bs1_sim.onnx
python -m onnxsim --input-shape="16,3,112,112" ./model/model_ir_se100_bs16.onnx ./model/model_ir_se100_bs16_sim.onnx
```

5.使用tensorRT工具测试onnx模型性能

```
./trtexec --onnx=model/model_ir_se100_bs1_sim.onnx --fp16 --shapes=image:1x3x112x112 --device=0
./trtexec --onnx=model/model_ir_se100_bs16_sim.onnx --fp16 --shapes=image:16x3x112x112 --device=0
```



### 3.2 onnx转om模型

1.设置环境变量

```bash
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

2.使用 atc 将 onnx 模型转换为 om 模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)

```bash
atc --framework=5 --model=./model/model_ir_se100_bs1_sim.onnx --output=model/model_ir_se100_bs1 --input_format=NCHW --input_shape="image:1,3,112,112" --log=debug --soc_version=Ascend310 

atc --framework=5 --model=./model/model_ir_se100_bs16_sim.onnx --output=model/model_ir_se100_bs16 --input_format=NCHW --input_shape="image:16,3,112,112" --log=debug --soc_version=Ascend310 
```



## 4 数据集预处理

### 4.1 数据集获取

获取LFW数据集，放在工作目录的data目录下

OBS： [lfw.bin](obs://l-resnet100e-ir/infer/lfw.bin) 云盘： [lfw.bin](https://drive.google.com/file/d/1mRB0A8f0b5GhH7w0vNMGdPjSWF-VJJLY/view?usp=sharing) 

```bash
mkdir data
mv lfw.bin ./data
```



### 4.2 数据集预处理

执行预处理脚本，会在工作目录的data目录下生成数据集预处理后的 bin 文件和 lfw 数据集标签

+ 参数1：'jpg'模式，功能是生成数据 bin 文件和 target 文件
+ 参数2：lfw数据集文件位置
+ 参数3：数据bin文件存放目录

```
python LResNet_preprocess.py 'jpg' './data/lfw.bin' './data/lfw'
```

### 4.3 生成预处理数据集信息文件

执行生成数据集信息脚本，会在工作目录下生成数据集信息文件

+ 参数1：'bin'模式，功能是生成数据集信息文件
+ 参数2：数据bin文件存放目录
+ 参数3：info文件存储路径
+ 参数4：宽
+ 参数5：高

```
python LResNet_preprocess.py 'bin' './data/lfw' './lfw.info' 112 112
```



## 5 离线推理

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)

### 5.2 离线推理

1.设置环境变量

```bash
export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/
```

2.执行离线推理, 输出结果默认保存在当前目录result/dumpOutput_device0

```bash
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./model/model_ir_se100_bs1.om -input_text_path=./lfw.info -input_width=112 -input_height=112 -output_binary=False -useDvpp=False

./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=16 -om_path=./model/model_ir_se100_bs16.om -input_text_path=./lfw.info -input_width=112 -input_height=112 -output_binary=False -useDvpp=False
```



## 6 精度对比

### 6.1 离线推理精度统计

执行后处理脚本统计om模型推理结果的Accuracy

+ 参数1：om模型预测结果目录
+ 参数2：lfw数据集target 文件

```shell
python LResNet_postprocess.py ./result/dumpOutput_device0 ./data/lfw_list.npy
```

控制台输出如下信息

```
accuracy: 0.9976666666666667
best_thresholds: 1.4140000000000001
```



### 6.2 开源精度

源代码仓公布精度

```
Model				Dataset		Accuracy
LResNet100E-IR 		LFW			0.998
```



### 6.3 精度对比

将得到的om离线模型推理Accuracy与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  



## 7 性能对比

### 7.1 npu性能数据

**1. batch_size=1**

```
[e2e] throughputRate: 78.7504, latency: 304761
[data read] throughputRate: 130.821, moduleLatency: 7.64405
[preprocess] throughputRate: 130.473, moduleLatency: 7.66442
[infer] throughputRate: 78.9751, Interface throughputRate: 82.9307, moduleLatency: 12.4253
[post] throughputRate: 78.975, moduleLatency: 12.6622
```

batch_size=1 Ascend310单卡吞吐率：82.9307*4=331.7228 fps



**2. batch_size=4**

```
[e2e] throughputRate: 127.206, latency: 188671
[data read] throughputRate: 140.878, moduleLatency: 7.09834
[preprocess] throughputRate: 140.464, moduleLatency: 7.11927
[infer] throughputRate: 127.885, Interface throughputRate: 133.476, moduleLatency: 7.72563
[post] throughputRate: 31.971, moduleLatency: 31.2783
```

batch_size=4 Ascend310单卡吞吐率：133.476*4=533.904 fps



**3. batch_size=8**

```
[e2e] throughputRate: 122.035, latency: 196665
[data read] throughputRate: 123.164, moduleLatency: 8.11925
[preprocess] throughputRate: 122.857, moduleLatency: 8.13957
[infer] throughputRate: 122.739, Interface throughputRate: 142.639, moduleLatency: 7.46171
[post] throughputRate: 15.3423, moduleLatency: 65.1795
```

batch_size=16 Ascend310单卡吞吐率：142.639*4=570.556 fps



**4. batch_size=16**

```
[e2e] throughputRate: 136.597, latency: 175699
[data read] throughputRate: 151.992, moduleLatency: 6.57929
[preprocess] throughputRate: 151.467, moduleLatency: 6.60209
[infer] throughputRate: 137.626, Interface throughputRate: 149.166, moduleLatency: 7.10235
[post] throughputRate: 8.6015, moduleLatency: 116.259
```

batch_size=16 Ascend310单卡吞吐率：149.166*4=596.664 fps



**5. batch_size=32**

```
[e2e] throughputRate: 125.799, latency: 190780
[data read] throughputRate: 127.155, moduleLatency: 7.86443
[preprocess] throughputRate: 126.743, moduleLatency: 7.89
[infer] throughputRate: 126.596, Interface throughputRate: 143.838, moduleLatency: 7.36579
[post] throughputRate: 3.95601, moduleLatency: 252.78
```

batch_size=16 Ascend310单卡吞吐率：143.838*4=575.352 fps



### 7.2 npu性能优化

OBS：[model_ir_se100_bs1.om](obs://l-resnet100e-ir/infer/model_ir_se100_bs1.om)  [model_ir_se100_bs16.om](obs://l-resnet100e-ir/infer/model_ir_se100_bs16.om) 云盘：[model_ir_se100_bs1.om](https://drive.google.com/file/d/1G9XFmmzmz5YJHN6RCrRiSRsfrQnfjZBM/view?usp=sharing)  [model_ir_se100_bs16.om](https://drive.google.com/file/d/1goyeBp3LZ_eai1fO01SofZXaHp4fcLB4/view?usp=sharing)

**1. batch_size=1**

```
[e2e] throughputRate: 78.7239, latency: 304863
[data read] throughputRate: 123.916, moduleLatency: 8.06996
[preprocess] throughputRate: 123.534, moduleLatency: 8.09494
[infer] throughputRate: 79.1652, Interface throughputRate: 83.2064, moduleLatency: 12.3888
[post] throughputRate: 79.165, moduleLatency: 12.6318
```

batch_size=1 Ascend310单卡吞吐率：83.2064*4=332.8256 fps

**2. batch_size=16**

```
[e2e] throughputRate: 129.278, latency: 185646
[data read] throughputRate: 130.232, moduleLatency: 7.67858
[preprocess] throughputRate: 129.894, moduleLatency: 7.69859
[infer] throughputRate: 129.767, Interface throughputRate: 186.532, moduleLatency: 5.76826
[post] throughputRate: 8.1103, moduleLatency: 123.3
```

batch_size=16 Ascend310单卡吞吐率：186.532*4=746.128 fps

### 7.3 npu性能优化前后对比

| batch size |  优化前  |  优化后  |
| :--------: | :------: | :------: |
|     1      | 331.7228 | 332.8256 |
|     16     | 596.664  | 746.128  |



### 7.4 gpu和npu性能对比

| batch size | GPU(FPS) | NPU(FPS) |
| :--------: | -------- | -------- |
|     1      | 241.5686 | 332.8256 |
|     16     | 678.984  | 746.128  |



