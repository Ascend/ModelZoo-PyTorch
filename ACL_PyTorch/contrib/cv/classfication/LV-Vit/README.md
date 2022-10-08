# LV-Vit ONNX模型端到端推理指导
- [LV-Vit ONNX模型端到端推理指导](#LV-Vit ONNX模型端到端推理指导)
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
    - [5 离线推理](#5-离线推理)
        - [5.1 benchmark工具概述](#51-benchmark工具概述)
        - [5.2 离线推理](#52-离线推理)
    - [6 精度对比](#6-精度对比)
        - [6.1 310P离线推理精度](#61-310P离线推理精度)
        - [6.2 开源精度](#62-开源精度)
        - [6.3 精度对比](#63-精度对比)
    - [7 性能对比](#7-性能对比)
        - [7.1 性能对比](#71-性能对比)
  

## 1 模型概述

### 1.1 论文地址

[LV-Vit论文](https://arxiv.org/abs/2104.10858 )

### 1.2 代码地址

[LV-Vit代码](https://github.com/zihangJiang/TokenLabeling )
branch=master
commit_id=2a217161fd5656312c8fac447fffbb6b3c091af7



## 2 环境说明

### 2.1 深度学习框架

```
CANN == 5.1.RC1
torch==1.8.0
torchvision==0.9.0
onnx==1.10.1
onnx-simplifier==0.3.6
```

### 2.2 python第三方库

```
numpy==1.21.2
pyyaml==5.4.1
pillow==8.3.1
timm==0.4.5
scipy==0.24.2
```



## 3 模型转换

### 3.1 pth转onnx模型

1.LV-Vit模型代码下载

```bash
# 切换到工作目录
cd LV-Vit

git clone https://github.com/zihangJiang/TokenLabeling.git
cd TokenLabeling
patch -p1 < ../LV-Vit.patch
cd ..
```

2.获取模型权重，并放在工作目录的model文件夹下
在model/下已经存放了在gpu8p上训练得到的pth，如需下载官方pth，则执行以下代码
```bash
wget https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_s-26M-224-83.3.pth.tar
mv lvvit_s-26M-224-83.3.pth.tar model_best.pth.tar

rm ./model/model_best.pth.tar
mv model_best.pth.tar ./model/
```



3.使用 LV_Vit_pth2onnx.py 脚本将pth模型文件转为onnx模型文件

+ 参数1：pth模型权重的路径

+ 参数2：onnx模型权重的存储路径

+ 参数3：batch size

```bash.
python LV_Vit_pth2onnx.py ./model/model_best.pth.tar ./model/model_best_bs1.onnx 1
```

4.使用 onnxsim 工具优化onnx模型

+ 参数1：输入的shape
+ 参数2：onnx模型权重的存储路径
+ 参数3：优化后onnx模型权重的存储路径

```
python -m onnxsim --input-shape="1,3,224,224" ./model/model_best_bs1.onnx ./model/model_best_bs1_sim.onnx
```



### 3.2 onnx转om模型

1.设置环境变量

```
source  /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.使用 atc 将 onnx 模型转换为 om 模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23710424%7C251366513%7C22892968%7C251168373)

```
atc --framework=5 --model=./model/model_best_bs1_sim.onnx --output=./model/model_best_bs1_sim --input_format=NCHW --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend${chip_name}

```
${chip_name}可通过`npu-smi info`指令查看，例：310P3

![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)   


## 4 数据集预处理

### 4.1 数据集获取

获取imagenet纯验证数据集，放在该目录：/opt/npu/imagenet/val/



### 4.2 数据集预处理

执行预处理脚本，会在工作目录的data目录下生成数据集预处理后的 bin 文件和 数据集信息文件

LV_Vit_preprocess.py：
+ --src_path: imagenet纯验证集路径; --save_path: bin文件存放路径

gen_dataset_info.py
+ 参数1：bin文件
+ 参数2：数据bin文件存放目录

```
python LV_Vit_preprocess.py --src_path /opt/npu/imagenet/PureVal/ --save_path ./data/prep_dataset;
python gen_dataset_info.py ./data/prep_dataset ./data/lvvit_prep_bin.info;
```


## 5 离线推理

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310P上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)

### 5.2 离线推理

1.设置环境变量

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.执行离线推理, 输出结果默认保存在当前目录result/dumpOutput_device0

```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./model/model_best_bs1_sim.om -input_text_path=lvvit_prep_bin.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False
```



## 6 精度对比

### 6.1 离线推理精度统计

执行后处理脚本统计om模型推理结果的Accuracy

+ 参数1：om模型预测结果目录
+ 参数2：imagenet纯验证集标签

```shell
python LV_Vit_postprocess.py ./result/dumpOutput_device0 ./data/val.txt
```

控制台输出如下信息

```
accuracy: 0.8317
```



### 6.2 开源精度

源代码仓公布精度

```
Model		Dataset		Accuracy
LV-Vit 		imagenet	 0.833
```



### 6.3 精度对比

将得到的om离线模型推理Accuracy与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。  



## 7 性能对比   

|         | 310    | 310P   | T4     | 310P/310 | 310P/T4 |
|---------|--------|--------|--------|----------|---------|
| bs1     | 179.04 | 180.93 | 160.77 | 1.01     | 1.12    |
| bs4     | 237.94 | 360.09 | 232.15 | 1.51     | 1.55    |
| bs8     | 242.20 | 407.14 | 239.95 | 1.68     | 1.69    |
| bs16    | 233.35 | 392.71 | 240.60 | 1.68     | 1.63    |
| bs32    | 227.57 | 323.18 | 243.34 | 1.42     | 1.32    |
| bs64    | 226.29 | 310.09 | 239.16 | 1.37     | 1.29       |
| 最优batch | 242.20 | 407.14 | 243.34 | 1.68     | 1.67        |


