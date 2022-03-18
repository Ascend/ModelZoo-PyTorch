## FOTS Onnx 模型 PyTorch 离线推理

### 1 模型概述

- 论文地址

```
https://arxiv.org/abs/1801.01671
```

- 代码地址

```
https://github.com/Wovchena/text-detection-fots.pytorch
```

- 数据集

```
下载使用ICDAR2015数据集：
解压后将ch4_test_images文件夹和gt.zip压缩标签文件放到根目录下
```

### 2 环境说明

```
CANN = 5.0.3
pytorch = 1.5.0
torchvision = 0.6.0
onnx = 1.7.0
numpy = 1.21.2
shapely = 1.6.4.post2(重要)
polygon3 = 3.0.9.1
opencv-python = 3.4.10.37(重要)
```

> X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip install 包名 安装
>
> Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip install 包名 安装



### 3 pth 转 om 模型

- pth 权重文件默认路径为根目录
- 进入根目录下执行 `./test/pth2onnx.sh` 脚本，自动生成生成 onnx 模型文件
- (执行./pth2onnx_8.sh脚本，生成batchsize=8的onnx模型文件)

```py
bash ./test/pth2onnx.sh 
```

- 执行 `./onnx2om.sh` 脚本，自动生成生成 om 模型文件
- （执行./onnx2om_8.sh脚本，生成batchsize=8的om模型文件）

```py
bash ./test/onnx2om.sh 
```


### 4 生成输入数据并保存为.bin文件

- 数据集默认路径为 `./ch4_test_images.zip` , 解压此数据集，在源码根目录下建立空文件夹用来保存预处理后的二进制图片，命名为res



- 使用脚本 `preprocess.sh`和`gen_dataset_info.sh` 获得预处理图片、二进制 bin 文件及其对应的路径信息

```
bash ./test/preprocess.sh
bash ./test/gen_dataset_info.sh
```


### 5 离线推理

####  5.1 benchmark工具概述

benchmark工具提供离线推理功能，输入 om 模型和模型所需要的输入 bin 文件，输出模型的输出数据文件。模型必须是通过 atc 工具转换的 om 模型，输入 bin 文件需要符合模型的输入要求。


####  5.2 离线推理

```
bash ./test/inference.sh
```
- (执行bash ./test/inference_8.sh脚本生成batchsize=8的二进制推理文件)


输出数据默认保存在根目录的 `./result/pref_visionbatchsize_1_device_0.txt` 中，可以看到时延和 FPS。输出图片默认保存在根目录的 `./result/dumpOutput_device0` 下.


### 6 精度对比

进入根目录下建立空文件夹用来保存后处理的坐标信息，命名为outPost。调用 ` postprocess.py` 来进行后处理，把输出的 bin 文件转换为对应坐标信息的txt文件。

```
python postprocess.py

```


- (执行 python postprocess_8.py输出batchsize=8推理的后处理文件)

详细的坐标信息结果在根目录的outPost/目录下，在根目录下建立空文件夹runs。调用 ` script.py` 来进行精度计算，将输出结果与真实标签比对。


```
zip -jmq runs/u.zip outPost/* && python3 script.py -g=gt.zip -s=runs/u.zip
```

### 7 性能对比

#### 7.1 NPU 310 性能数据
```
(310 bs1) Inference average time: 9.9045 ms
(310 bs1) FPS:39.618
```

根据时延和核心数，计算得到 Batchsize = 1 时单卡吞吐率 39.618 FPS

```
(310 bs8) Inference average time: 9.3025 ms
(310 bs8) FPS:37.210
```

根据时延和核心数，计算得到 Batchsize = 8 时单卡吞吐率 37.210 FPS

#### 7.2 GPU T4 性能数据


根据时延和核心数，计算得到 Batchsize = 1 时单卡吞吐率 44.704 FPS


根据时延和核心数，计算得到 Batchsize = 8 时单卡吞吐率 47.271 FPS

#### 7.3 性能对比

| Batch Size | 310 (FPS/Card) | T4 (FPS/Card) | 310/T4   |
| ---------- | -------------- | ------------- | -------- |
| 1          | *39.618*       | *44.704*      | *88.62%* |
| 8          | *37.210*       | *47.271*      | *78.71%* |
