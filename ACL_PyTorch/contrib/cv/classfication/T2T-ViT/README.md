# T2T-ViT Onnx模型端到端推理指导

## 1 模型概述


### 1.1 论文地址

[Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://arxiv.org/abs/2101.11986)

### 1.2 代码地址

开源仓：[https://github.com/yitu-opensource/T2T-ViT](https://github.com/yitu-opensource/T2T-ViT)<br>
branch：main<br>
commit_id：0f63dc9558f4d192de926504dbddfa1b3f5db6ca<br>

## 2 环境说明

该模型离线推理使用 Atlas 300I Pro 推理卡，所有步骤都在 [CANN 5.1.RC1](https://www.hiascend.com/software/cann/commercial) 环境下进行，CANN包以及相关驱动、固件的安装请参考 [软件安装](https://www.hiascend.com/document/detail/zh/canncommercial/51RC1/envdeployment/instg)。
### 2.1 安装依赖
```shell
conda create -n ${env_name} python=3.7.5
conda activate ${env_name}
pip install -r requirements.txt 
```

### 2.2 获取开源仓代码
```shell
git clone https://github.com/yitu-opensource/T2T-ViT.git
cd T2T-ViT
git checkout main
git reset --hard 0f63dc9558f4d192de926504dbddfa1b3f5db6ca
```

## 3 源码改动
为什么改动，怎样改动？


## 4 模型转换

### 4.1 Pytorch转ONNX模型

1. 下载pth权重文件
```shell
wget https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.5_T2T_ViT_14.pth.tar
```

2. 执行T2T_ViT_pth2onnx.py脚本，生成ONNX模型文件

```shell
python3.7 T2T_ViT_pth2onnx.py --
```

### 4.2 ONNX转OM模型

1. 设置环境变量

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

该命令中使用CANN默认安装路径(/usr/local/Ascend/ascend-toolkit)中的环境变量，使用过程中请按照实际安装路径设置环境变量。

2、生成OM模型
ATC工具的使用请参考 [ATC模型转换](https://www.hiascend.com/document/detail/zh/canncommercial/51RC1/inferapplicationdev/atctool)

```shell
atc --framework=5 --model=${onnx-path} --output=${om-path} --input_format=NCHW --input_shape="image:${bs},3,224,224" --log=error --soc_version=Ascend${chip_name} --keep_dtype=keep_dtype.cfg
```
说明：<br>
--model 指定ONNX模型的路径<br>
--output 生成OM模型的保存路径<br>
执行命令前，需设置--input_shape参数中bs的数值，例如：1、4、8、16 <br> 
chip_name可通过`npu-smi info`指令查看，例：310P3<br>
![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

## 5 数据预处理


### 5.1 数据集获取

该模型使用[ImageNet官网](http://www.image-net.org/)的5万张验证集进行测试
数据集结构如下：
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### 5.2 数据集预处理

1. 生成预处理脚本T2T_ViT_preprocess.py
```shell
补充
```

2. 执行预处理脚本，生成数据集预处理后的bin文件

```shell
python3.7 T2T_ViT_preprocess.py -–data-dir ${dataset_path} --out-dir ${prep_output_dir} –gt-path ${groundtruth_path} -–batch-size ${batchsize}
```
参数说明：


## 6 离线推理

### 6.1 msame工具

本项目使用msame工具进行推理，msame编译及用法参考[msame推理工具](https://gitee.com/ascend/tools/tree/master/msame)。<br>
msame推理前需要设置环境变量：
``` shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
```

该命令中使用CANN默认安装路径(/usr/local/Ascend/ascend-toolkit)中的环境变量，使用过程中请按照实际安装路径设置环境变量。

### 6.2 离线推理

运行如下命令进行离线推理：

```shell
./msame --model ${om_path}  --input ${input_dataset_path} --output ${output_dir} --outfmt BIN
```
参数说明：


输出结果保存在output_dir中，文件类型为bin文件。

### 6.3 精度验证

1.生成后处理脚本T2T_ViT_postprocess.py脚本
```shell
补充
```

2.运行T2T_ViT_postprocess.py脚本并与npy文件比对，可以获得Accuracy Top1数据。

```shell
python3.7 T2T_ViT_postprocess.py –result-dir ${msame_bin_path} –gt-path ${gt_path} --batch-size ${batchsize}
```
参数说明：<br>


### 6.4 性能验证
用msame工具进行纯推理100次，然后根据平均耗时计算出吞吐率。
```shell

```
说明：


## 7 精度和性能对比

总结

各batchsize对比结果如下：

|     模型     |                        开源仓Pytorch精度                        | 310P离线推理精度 | 基准性能 | 310P性能 |
| :----------: | :-------------------------------------------------------: | :--------------: | :------: | :------: |
| T2T-ViT bs1  | [rank1:81.5%](https://github.com/yitu-opensource/T2T-ViT) |   rank1:81.4%    |  24fps   |  142fps  |
| T2T-ViT bs4  | [rank1:81.5%](https://github.com/yitu-opensource/T2T-ViT) |   rank1:81.4%    |  32fps   |  179fps  |
| T2T-ViT bs8  | [rank1:81.5%](https://github.com/yitu-opensource/T2T-ViT) |   rank1:81.4%    |  39fps   |  212fps  |
| T2T-ViT bs16 | [rank1:81.5%](https://github.com/yitu-opensource/T2T-ViT) |   rank1:81.4%    |  35fps   |  210fps  |
| T2T-ViT bs32 | [rank1:81.5%](https://github.com/yitu-opensource/T2T-ViT) |   rank1:81.4%    |  34fps   |  203fps  |
| T2T-ViT bs64 | [rank1:81.5%](https://github.com/yitu-opensource/T2T-ViT) |   rank1:81.4%    |  36fps   |  198fps  |