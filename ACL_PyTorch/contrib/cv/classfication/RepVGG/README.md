# RepVGG模型PyTorch离线推理指导

## 1. 模型概述

[论文地址](https://arxiv.org/abs/2101.03697)

[代码地址](https://github.com/DingXiaoH/RepVGG)

## 2. 环境准备 

### 2.1 环境说明

```shell
torch == 1.8.0
torchversion == 0.9.0
onnx == 1.8.1
numpy == 1.20.2
Pillow == 7.2.0
opencv-python ==4.5.2.52
```

### 2.2 环境安装

#### 2.2.1 安装必要的依赖

测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装。

```shell
pip3 install -r requirements.txt  
```

#### 2.2.2 获取，修改与安装开源模型代码  

```shell
git clone https://github.com/DingXiaoH/RepVGG   
cd RepVGG  
git reset 9f272318abfc47a2b702cd0e916fca8d25d683e7 --hard
cd ..  
```

#### 2.2.3 获取权重文件  

[RepVGG-A0-train.pth](https://pan.baidu.com/s/1nCsZlMynnJwbUBKn0ch7dQ), access code "rvgg"

#### 2.2.4 获取数据集     

该模型使用ImageNet官网的5万张验证集进行测试，图片与标签分别存放在/opt/npu/imagenet/val与/opt/npu/imagenet/val_label.txt。

#### 2.2.5 获取benchmark工具

[获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer) ，将benchmark.x86_64或benchmark.aarch64放到当前目录  

## 3. 数据预处理 

### 3.1 数据集预处理

将数据集处理为可输入推理工具benchmark的格式
```shell
Python3.7 RepVGG_preprocess.py repvgg /opt/npu/imagenet/val ./prep_dataset
```

参数说明

- `参数1`：模型名
- `参数2`：数据集位置
- `参数3`：输出文件夹位置

### 3.2 生成数据集信息文件

将处理好的数据生成对应的info文件，作为benchmark工具推理的输入
```shell
python3 gen_dataset_info.py bin ./prep_dataset ./RepVGG_prep_bin.info 224 224
```

参数说明

- `参数1`：前一步处理好数据的文件格式
- `参数2`：数据位置
- `参数3`：生成info文件名称
- `参数4、5`：每张图片的宽和高

## 4. 模型转换

### 4.1 执行pth2onnx脚本生成onnx文件

```shell
python3 RepVGG_pth2onnx.py PepVGG-A0-train.pth RepVGG.onnx
```

参数说明

- `参数1`: 输入的权重文件（.pth）路径。
- `参数2`：输出的文件名。 

### 4.2 onnx模型转om模型

#### 4.2.1 设置环境变量

```shell
source ${HOME}/ascend-toolkit/set_env.sh
```

说明
设置CANN的环境变量，其中${HOME}为CANN包安装路径，默认在/usr/local/Ascend下

#### 4.2.2 使用ATC命令将onnx模型转换为om模型

执行前使用`npu-smi info`查看设备状态，确保device空闲

```shell
atc --framework=5 --model=RepVGG.onnx --output=RepVGG_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" \
    --log=error --soc_version=Ascend${chip_name}
```

参数说明

- `--model`: 输入的onnx模型路径。
- `--output`:输出的文件名。 
- `--input_format`: 输入形状的格式。
- `--input_shape`: 模型输入的形状。
- `--log`: 设置ATC模型转换过程中日志的级别
- `--soc_version`:目标芯片类型，如Ascend310、Ascend310P3，${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)


## 5 benchmark推理

### 5.1 benchmark工具简介

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310和Ascend310P上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程。

### 5.2 设置环境变量

```shell
source ${HOME}/ascend-toolkit/set_env.sh

```

说明
设置CANN的环境变量，其中${HOME}为CANN包安装路径，默认在/usr/local/Ascend下

### 5.3 执行推理

执行时使用`npu-smi info`查看设备状态，确保device空闲

```shell
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=RepVGG_bs1.om -input_text_path=./RepVGG_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

```

参数说明

- `-model_type`:benchmark支持的模型类型，目前支持的有vision，nmt，widedeep，nlp，yolocaffe，bert，deepfm
- `-device_id`:运行在ascend 310或ascend 310P的哪个device上，每张ascend 310卡有4个device，每张ascend 310P卡有1个device
- `-batch_size`:om模型的batch大小，该值应与om模型的batch大小相同，否则报输入大小不一致的错误
- `-om_path`:om模型文件路径
- `-input_text_path`:包含数据集每个样本的路径与其相关信息的数据集信息文件路径
- `-input_height`:输入高度
- `-input_width`:输入宽度
- `-output_binary`:以预处理后的数据集为输入，benchmark工具推理om模型的输出数据保存为二进制还是txt，但对于输出是int64类型的节点时，指定输出为txt时会将float类型的小数转换为0而出错
- `-useDvpp`:是否使用aipp进行数据集预处理

## 6. 精度和性能对比

### 6.1 离线推理精度

运行如下脚本评测精度，运行后精度结果保存在result.json文件中
```shell
python3 RepVGG_postprocess.py result/dumpOutput_device0/ /opt/npu/imagenet/val_label.txt ./result.json

```

参数说明

- `参数1`：benchmark推理结果文件所在路径
- `参数2`：数据集标注标签路径
- `参数3`：结果保存路径

### 7.2 npu性能数据

benchmark工具推理后生成result/perf_vision_batchsize_1_device_0.txt文件，其中Interface turoughputRate即为310P单卡吞吐量，Interface turoughputRate *4为310单卡吞吐量，运行脚本计算310P单卡吞吐量，计算结果回显

```shell
python3 test/parse.py result/perf_vision_batchsize_1_device_0.txt 310P

```

参数说明

- `参数1`：benchmark性能统计文件路径
- `参数2`：目标芯片名，目前支持310P、310

## 8. 评测结果：

|    模型     |                    官网pth精度                     | 310P离线推理精度 | T4基准性能 | 310性能 | 310P性能 |
| :---------: | :------------------------------------------------: | :--------------: | :--------: | ------: | :------: |
| RepVGG bs1  | [acc1:72.418](https://github.com/DingXiaoH/RepVGG) |    acc1:72.15    |  1128fps   | 1112fps | 1919fps  |
| RepVGG bs16 | [acc1:72.418](https://github.com/DingXiaoH/RepVGG) |    acc1:72.15    |  4784fps   | 7115fps | 8752fps  |
