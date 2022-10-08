# CenterFace Onnx模型端到端推理指导

- 1 模型概述
  - [1.1 论文地址](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#11-论文地址)
  - [1.2 代码地址](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#12-代码地址)
- 2 环境说明
  - [2.1 深度学习框架](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#21-深度学习框架)
  - [2.2 python第三方库](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#22-python第三方库)
- 3 模型转换
  - [3.1 pth转onnx模型](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#31-pth转onnx模型)
  - [3.2 onnx转om模型](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#32-onnx转om模型)
- 4 数据集预处理
  - [4.1 数据集获取](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#41-数据集获取)
  - [4.2 数据集预处理](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#42-数据集预处理)
  - [4.3 生成数据集信息文件](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#43-生成数据集信息文件)
- 5 离线推理
  - [5.1 benchmark工具概述](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#51-benchmark工具概述)
  - [5.2 离线推理](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#52-离线推理)
- 6 精度对比
  - [6.1 离线推理精度统计](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#61-离线推理精度统计)
  - [6.2 开源精度](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#62-开源精度)
  - [6.3 精度对比](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#63-精度对比)
- 7 性能对比
  - [7.1 npu性能数据](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#71-npu性能数据)
  - [7.2 T4性能数据](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#72-T4性能数据)
  - [7.3 性能对比](https://gitee.com/ascend/modelzoo/tree/master/built-in/ACL_PyTorch/Benchmark/cv/classification/ResNext50#73-性能对比)
- 8 310P增加文件介绍

## 1 模型概述

- **论文地址**
- **代码地址**

### 1.1 论文地址

[CenterFace论文](https://arxiv.org/abs/1911.03599)

### 1.2 代码地址

[CenterFace代码](https://github.com/chenjun2hao/CenterFace.pytorch)

## 2 环境说明

- **深度学习框架**
- **python第三方库**

### 2.1 深度学习框架

```
python3.7.5
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

## 3 模型转换

- **pth转onnx模型**
- **onnx转om模型**

### 3.1 pth转onnx模型

1.下载pth权重文件
权重文件从百度网盘上获取：https://pan.baidu.com/s/1sU3pRBTFebbsMDac-1HsQA        密码：etdi

2.使用pth2onnx.py进行onnx的转换

```
mv ./CenterFace/center-face/src/pth2onnx.py  ./CenterFace/center-face/src/lib
cd ./CenterFace/center-face/src/lib
python3 pth2onnx.py
```

### 3.2 onnx转om模型

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN 5.0.1 开发辅助工具指南 (推理) 01

```
cd ./CenterFace/center-face/src/test
bash onnxToom.sh
```

## 4 数据集预处理

- **数据集获取**
- **数据集预处理**
- **生成数据集信息文件**

### 4.1 数据集获取

拉取代码仓库 （因为使用了开源代码模块，所以需要git clone一下）

```shell
git clone https://gitee.com/Levi990223/center-face.git
```

整理代码结构

```shell
mv -r test center-face/src
mv benchmark.x86_64 centerface_pth_preprocess.py centerface_pth_postprocess.py convert.py CenterFace.onnx pth2onnx.py get_info.py model_best.pth move.sh README.md ./center-face/src
```

下载WIDER_FACE数据集，将图片上在这个目录下：

下载地址：https://www.graviti.cn/open-datasets/WIDER_FACE

```
$CenterFace_ROOT/center-face/data/{eval_dataset}
```

### 4.2 数据集预处理

1.预处理脚本centerface_pth_preprocess.py

2.执行预处理脚本，生成数据集预处理后的bin文件

```
cd ./CenterFace/center-face/src/test 
bash start.sh
```

### 4.3 生成数据集信息文件

1.生成数据集信息文件脚本get_info.py

2.执行生成数据集信息脚本，生成数据集信息文件

```
cd ./CenterFace/center-face/src/test
bash to_info.sh
```

to_info.sh里，第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息

## 5 离线推理

- **benchmark工具概述**
- **离线推理**

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN 5.0.1 推理benchmark工具用户指南 01

### 5.2 离线推理

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.执行离线推理

执行前需要将benchmark.x86_64移动到执行目录下

(注：执行目录是/center-face/src)

然后运行如下命令：

```
cd ./CenterFace/center-face/src/test
bash infer.sh
```

输出结果默认保存在当前目录result/dumpOutput_device{0}，每个输入对应的输出对应四个_x.bin文件。

3.处理目录result/dumpOutput_device{0}下的bin文件

将该目录下的文件分类别存放，以便于后处理

```
cd ./CenterFace/center-face/src/
python3 convert.py ./result/dumpOutput_device1/ ./result/result
```

第一个参数是benchmark得到的bin文件目录，第二个参数是保存路径

## 6 精度对比

- **离线推理精度**
- **开源精度**
- **精度对比**

### 6.1 离线推理精度统计

1.后处理

注：这里需要使用wide_face_val.mat文件，在center-face/evaluate/ground_truth/可以找到，然后将其移动到center-face/src目录下,然后执行下面命令

```
cd ./CenterFace/center-face/src
python3 centerface_pth_postprocess.py
```

2.进行Ascend310上精度评估

```
cd ./CenterFace/center-face/evaluate
python3 evaluation.py
```

### 6.2 开源精度

[CenterFace官网精度]([chenjun2hao/CenterFace.pytorch: unofficial version of centerface, which achieves the best balance between speed and accuracy at face detection (github.com)](https://github.com/chenjun2hao/CenterFace.pytorch))

```
Easy   Val AP: 0.9257383419951156
Medium Val AP: 0.9131308732465665
Hard   Val AP: 0.7717305552550734
```

### 6.3 精度对比

```
Easy   Val AP: 0.9190736484158941
Medium Val AP: 0.9067769085346155
Hard   Val AP: 0.7425807072008017
```

### 6.3 精度对比

实际上官网的hard精度达不到77%，最高74%左右，所以对比下来精度是达标的。

## 7 性能对比

- **npu性能数据**
- **T4性能数据**
- **性能对比**

### 7.1 npu性能数据

1.benchmark工具在整个数据集上推理获得性能数据
batch1的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_1_device_0.txt：

```
[e2e] throughputRate: 33.1307, latency: 97372
[data read] throughputRate: 36.336, moduleLatency: 27.5209
[preprocess] throughputRate: 35.6065, moduleLatency: 28.0847
[infer] throughputRate: 33.4556, Interface throughputRate: 91.86, moduleLatency: 29.2697
[post] throughputRate: 33.4544, moduleLatency: 29.8915
```

Interface throughputRate: 91.86，91.86x4=367.44既是batch1 310单卡吞吐率
batch16的性能，benchmark工具在整个数据集上推理后生成result/perf_vision_batchsize_16_device_1.txt：

```
[e2e] throughputRate: 31.7581, latency: 101580
[data read] throughputRate: 35.0206, moduleLatency: 28.5547
[preprocess] throughputRate: 33.9534, moduleLatency: 29.4521
[infer] throughputRate: 32.022, Interface throughputRate: 80.3537, moduleLatency: 30.4381
[post] throughputRate: 2.00424, moduleLatency: 498.943

```

Interface throughputRate: 80.3537，80.3537x4=321.4148既是batch16 310单卡吞吐率

### 7.2 T4性能数据

```
[W] [TRT] TensorRT was linked against cuBLAS/cuBLAS LT 11.2.0 but loaded cuBLAS/cuBLAS LT 11.1.0
[W] [TRT] TensorRT was linked against cuBLAS/cuBLAS LT 11.2.0 but loaded cuBLAS/cuBLAS LT 11.1.0
[W] [TRT] TensorRT was linked against cuBLAS/cuBLAS LT 11.2.0 but loaded cuBLAS/cuBLAS LT 11.1.0
t4 bs1 fps:337.544
[W] [TRT] TensorRT was linked against cuBLAS/cuBLAS LT 11.2.0 but loaded cuBLAS/cuBLAS LT 11.1.0
[W] [TRT] TensorRT was linked against cuBLAS/cuBLAS LT 11.2.0 but loaded cuBLAS/cuBLAS LT 11.1.0
[W] [TRT] TensorRT was linked against cuBLAS/cuBLAS LT 11.2.0 but loaded cuBLAS/cuBLAS LT 11.1.0
t4 bs16 fps:359.999
```

batch1 t4单卡吞吐率：337.544

batch16 t4单卡吞吐率：359.999

### 7.3 性能对比

batch1：91.86x4=367.44 > 337.544
batch16：80.3537x4=321.4148 < 359.999

## 8 310P增加文件介绍

1.aipp_centerface.aippconfig ONNX模型转OM模型时所配置aipp
2.calibration_bin.py 量化模型时输入真实数据的组件脚本 
