- [1. T2T-ViT Onnx模型端到端推理指导](#1-T2T-ViT-onnx模型端到端推理指导)
  - [1.1. 模型概述](#11-模型概述)
  	- [1.1.1. 论文地址](#111-论文地址)
  	- [1.1.2. 代码地址](#112-代码地址)
  - [1.2. 环境说明](#12-环境说明)
  - [1.3. 模型转换](#13-模型转换)
  	- [1.3.1. pth转onnx模型](#131-pth转onnx模型)
  	- [1.3.2. onnx转om模型](#132-onnx转om模型)
  - [1.4. 数据预处理](#14-数据预处理)
  	- [1.4.1. 数据集获取](#141-数据集获取)
  	- [1.4.2. 1.4.2.数据集预处理](#142-142数据集预处理)
  - [1.5. 离线推理](#15-离线推理)
  	- [1.5.1. msame工具概述](#151-msame工具概述)
  	- [1.5.2.  离线推理](#152--离线推理)
  - [1.6. 精度和性能对比](#16-精度和性能对比)
  	- [1.6.1. 离线推理TopN精度](#161-离线推理topn精度)
  	- [1.6.2. 开源TopN精度](#162-开源topn精度)
  	- [1.6.3. 对比结果](#163-对比结果)

# 1. T2T-ViT Onnx模型端到端推理指导

## 1.1. 模型概述


### 1.1.1. 论文地址

[T2T-ViT论文](https://arxiv.org/abs/2101.11986)

### 1.1.2. 代码地址

[T2T-ViT代码](https://github.com/yitu-opensource/T2T-ViT)
commit_id=143df41f2e372364188027d826393cdff99a37fd

## 1.2. 环境说明


```
CANN 5.1.RC1
torch==1.5.0+ascend.post5.20220315
torchvision==0.6.0
timm==0.3.2
```

 **说明：**

- X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
- Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装
- preprocess.py、pth2onnx.py、postprocess.py需要配合[原仓库代码](https://github.com/yitu-opensource/T2T-ViT)使用。

## 1.3. 模型转换

- [T2T-ViT预训练pth、onnx、om权重文件，提取码：y3b2](https://pan.baidu.com/s/1PPRgM_UQlOoG9twagTjBEA)

### 1.3.1. pth转onnx模型

1. 下载pth权重文件
   [T2T-ViT预训练pth、onnx、om权重文件，提取码：y3b2](https://pan.baidu.com/s/1PPRgM_UQlOoG9twagTjBEA)

2. T2T-ViT模型代码在models文件里
4. 执行pth2onnx.py脚本，生成onnx模型文件

```
python3.7 pth2onnx.py
```

### 1.3.2. onnx转om模型

1. 设置环境变量

```
source env.sh
```

2. 使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)

```

# 将atc日志打印到屏幕
#export ASCEND_SLOG_PRINT_TO_STDOUT=1
# 设置日志级别
#export ASCEND_GLOBAL_LOG_LEVEL=0 #debug 0 --> info 1 --> warning 2 --> error 3
# 开启ge dump图
#export DUMP_GE_GRAPH=2
# 参考命令
atc --framework=5 --model=T2T_ViT_14.onnx --output=T2T_ViT_14_bs1_test --input_format=NCHW --input_shape="input:1,3,224,224" --soc_version=Ascend310 --keep_dtype=keep_dtype.cfg
```

若生成batch size为1的om模型，对应的命令为：

```
atc --framework=5 --model=T2T_ViT_14.onnx --output=T2T_ViT_14_bs1_test --input_format=NCHW --input_shape="input:1,3,224,224" --soc_version=Ascend310 --keep_dtype=keep_dtype.cfg
```

batch size为4、8、16、32、64的同上

## 1.4. 数据预处理


### 1.4.1. 数据集获取

> 该模型使用[ImageNet官网](http://www.image-net.org/)的5万张验证集进行测试

### 1.4.2. 1.4.2.数据集预处理

1. 编写预处理脚本preprocess.py
   预处理方式有两种：不使用aipp的二进制输入，以及使用aipp的jpg输入，这里使用第一种。
2. 执行预处理脚本，生成数据集预处理后的bin文件

```
python3.7 preprocess.py -–data-dir ${dataset_path} --out-dir ${prep_output_dir} –-gt-path ${groundtruth_path}
```

## 1.5. 离线推理

### 1.5.1. msame工具概述

将获取的工具包并解压，将msame工具放在当前目录下。

### 1.5.2.  离线推理

1. 设置环境变量

``` 
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
```

2. 执行离线推理
   运行如下命令进行离线推理：

```
./msame --model ${om_path}  --input ${input_dataset_path} --output ${output_dir} --outfmt BIN
```

输出结果保存在output中，文件类型为bin文件。

## 1.6. 精度和性能对比

### 1.6.1. 离线推理TopN精度

后处理与精度统计

调用postprocess.py脚本与gt_bs1.npy比对，可以获得Accuracy Top1，Top5数据。

```
python3.7 postprocess.py –-result-dir ${msame_bin_path} –-gt-path ${gt_path}
```

查看输出的结果：

```
acc1:0.8051, acc5:0.9522
```


### 1.6.2. 开源TopN精度

GPU上使用[原仓库代码](https://github.com/yitu-opensource/T2T-ViT)对pth文件进行推理

得到的结果是：

```
python main.py path/to/data --model t2t_vit_14 -b 100 --eval_checkpoint path/to/pth
Top-1 accuracy of the model is: 81.5%
```

### 1.6.3. 对比结果

|    模型     |                        官网pth精度                        | 310离线推理精度 | 基准性能 | 310P性能 |
| :---------: | :-------------------------------------------------------: | :-------------: | :------: | :-----: |
| T2T-ViT bs1 | [rank1:81.5%](https://github.com/yitu-opensource/T2T-ViT) |   rank1:80.5%   |  24fps   | 153fps  |
| T2T-ViT bs8 | [rank1:81.5%](https://github.com/yitu-opensource/T2T-ViT) |   rank1:80.5%   |  39fps   | 270fps  |