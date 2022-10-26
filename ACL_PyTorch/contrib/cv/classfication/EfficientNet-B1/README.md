# EfficientNet-B1模型PyTorch离线推理指导

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
	-   [4.2 数据集切分](#42-数据集切分)
	-   [4.3 数据集预处理](#43-数据集预处理)
	-   [4.4 生成数据集信息文件](#44-生成数据集信息文件)
-   [5 离线推理](#5-离线推理)
	-   [5.1 ais_infer工具概述](#51-ais_infer工具概述)
	-   [5.2 离线推理](#52-离线推理)
-   [6 精度对比](#6-精度对比)
	-   [6.1 精度数据](#61-精度数据)
	-   [6.2 精度对比](#62-精度对比)
-   [7 性能对比](#7-性能对比)
	-   [7.1 310P性能数据](#71-310P性能数据)  
	-   [7.2 T4性能数据](#72-T4性能数据) 
	-   [7.3 性能对比](#73-性能对比)




## 1 模型概述

### 1.1 论文地址

[EfficientNet-B1](https://arxiv.org/abs/1905.11946)*是针对网络参数扩张方式问题，经过EfficientNet-B0进行扩张提高网络性能的网络。*

### 1.2 代码地址

[EfficientNet-B1 Pytorch实现代码](https://github.com/facebookresearch/pycls)

```sh
branch=master
commit_id=8c79a8e2adfffa7cae3a88aace28ef45e52aa7e5
```





## 2 环境说明

### 2.1 深度学习框架

```sh
CANN == 5.1.RC1
pytorch == 1.8.0
torchvision == 0.9.0
onnx == 1.10.0
```

### 2.2 python第三方库

```sh
onnx-simplifier == 0.4.5
isort==4.3.21
iopath
fairscale
flake8
pyyaml
matplotlib
numpy 
opencv-python
parameterized
setuptools
simplejson
submitit
yacs
yattag
scipy
decorator
sympy
```

安装必要的依赖，测试环境可能已经安装其中的一些不同版本的库了，故手动测试时不推荐使用该命令安装。

```sh
pip install -r requirements.txt
```





## 3 模型转换

### 3.1 pth转onnx模型

1.获取pth权重文件

```sh
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E4%BA%A4%E4%BB%98%E4%BB%B6/cv/image_classification/EfficientNet-B1/EN-B1_dds_8gpu.pyth
```

2.获取EfficientNet-B1源码

```sh
git clone https://github.com/facebookresearch/pycls
cd pycls
git reset f20820e01eef7b9a47b77f13464e3e77c44d5e1f --hard
cd ..
```

3.使用Efficient-B1_pth2onnx.py进行onnx的转换，在目录下生成Efficient-b1.onnx。

```sh
python3.7 Efficient-B1_pth2onnx.py
```

4.对onnx模型进行onnxsim优化。（以batch_size=16为例。）
```sh
python3.7 -m onnxsim --input-shape="image:16,3,240,240" ./Efficient-b1.onnx bs16_onnxsim.onnx
```

### 3.2 onnx转om模型

1.设置环境变量，请以实际安装环境配置。

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.使用atc将onnx模型转换为om模型文件，工具使用方法可以参考[CANN V100R020C10 开发辅助工具指南 (推理) 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164868?idPath=23310P424|251366513|22892968|251168373)，需要指定输出节点以去除无用输出，可以使用netron开源可视化工具查看具体的输出节点名：

```sh
atc --framework=5 --model=./bs16_onnxsim.onnx --input_format=NCHW --input_shape="image:16,3,240,240" --output=Efficient-b1_bs16 --log=debug --soc_version=Ascend${chip_name} --enable_small_channl=1
# 此处以bs=16为例。
# ${chip_name}可通过npu-smi info指令查看
```

![输入图片说明](https://images.gitee.com/uploads/images/2022/0704/095450_881600a3_7629432.png "屏幕截图.png")

参数说明： --model：为ONNX模型文件。 

--framework：5代表ONNX模型。 

--output：输出的OM模型。 

--input_format：输入数据的格式。 

--input_shape：输入数据的shape。 

--log：日志级别。

 --soc_version：处理器型号。





## 4 数据集预处理

### 4.1 数据集获取

获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的meta.mat。

数据目录结构请参考：

```
├──ILSVRC2012_img_val
├──val
├──ILSVRC2012_devkit_t12
     ├── data
           └── meta.mat
```

### 4.2 数据集切分

切分官方val数据集，形成和train一样的文件结构，即根目录-类别-图片 三级

```sh
# 第一个参数为 新下载且未分类的 imagenet的val数据集路径，
# 第二个参数为官方 提供的 devkit 文件夹，如果要保留val文件夹请先备份
python3.7 ImageNet_val_split.py ./val ./ILSVRC2012_devkit_t12
```

### 4.3 数据集预处理

执行“Efficient-B1_preprocess.py”预处理脚本，生成数据集预处理后的bin文件。

```sh
python3.7 Efficient-B1_preprocess.py ../val ./prep_dataset
```

### 4.4 生成数据集信息文件

生成数据集info文件。

```sh
python3.7 gen_dataset_info.py bin ./prep_dataset ./efficientnet-B1_prep_bin.info 240 240 
```





## 5 离线推理

### 5.1 ais_infer工具概述

ais_infer工具包含前端和后端两部分。 后端基于c+开发，实现通用推理功能； 前端基于python开发，实现用户界面功能。获取工具及使用方法可以参考[tools: Ascend tools - Gitee.com](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)

### 5.2 离线推理

1.设置环境变量，请以实际安装环境配置。

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.安装ais_infer工具

```sh
# 将工具的相关代码下载到本地
git clone https://gitee.com/ascend/tools.git
# 进入ais-bench/tool/ais_infer目录，执行如下命令进行编译，生成推理后端whl包
cd tool/ais_infer/backend/
pip3.7 wheel ./
# 在运行设备上执行如下命令，进行安装
pip3 install ./aclruntime-0.0.1-cp37-cp37m-linux_aarch64.whl
# 如果安装提示已经安装了相同版本的whl，请执行命令请添加参数"--force-reinstall"
```

3.建立软链接

```sh
# 由于“标签/bin文件”这一文件结构并不符合输出性能数据的要求，且直接cp文件量过多，因此选择建立软链接。
find ../prep_dataset/ -name "*.bin" | xargs -i ln -sf {} ./ruanlianjie
```

4.输出性能数据

```sh
# 以bs=16为例。
python3.7 ais_infer.py --model Efficient-b1_bs16.om --output ./ --input ./ruanlianjie --outfmt TXT
# --model 需要进行推理的om模型
# --output 推理数据输出路径
# --input 模型需要的输入，支持bin文件和目录，若不加该参数，会自动生成都为0的数据
# --outfmt 输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”
```

5.切换到测试性能时生成的{20xx_xx_xx-xx_xx_xx}文件夹下，并删除sumary.json

```sh
rm -rf sumary.json
```

6.对性能的输出结果进行重命名，以便于输出精度。

```sh
# 由于性能输出的文件以“0.txt”结尾，不符合输出精度所要求的“以1.txt结尾”这一要求，因此需要对输出精度时输出的文件进行重命名。

# 安装rename，已经安装则可以跳过。
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install rename
rename "s/_0.txt/_1.txt/" *
```



## 6 精度对比

### 6.1 精度数据

调用Efficient-B1_postprocess.py:

```sh
# 输出精度
python3.7 Efficient-B1_postprocess.py --pre_dir {20xx_xx_xx-xx_xx_xx} --data_dir ../val/ --save_file ./result.json
# 其中--pre_dir这一参数需要适时调整为测试性能时的输出的{20xx_xx_xx-xx_xx_xx}文件夹。
```

### 6.2 精度对比

|          | **310** | **310P** |
| -------- | ------- | -------- |
| **Top1** | 75.936% | 75.936%  |
| **Top5** | 92.774% | 92.774%  |

### 



## 7 性能对比

### 7.1 310P性能数据

测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务。

```sh
# 以bs=16为例。
python3.7 ais_infer.py --model Efficient-b1_bs16.om --output ./ --input ./ruanlianjie --outfmt TXT
# --model 需要进行推理的om模型
# --output 推理数据输出路径
# --input 模型需要的输入，支持bin文件和目录，若不加该参数，会自动生成都为0的数据
# --outfmt 输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”
```

Interface throughputRate: 1635.3即是batch16 310P单卡吞吐率。

### 7.2 T4性能数据

在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务。

```sh
trtexec --onnx=./Efficient-b1.onnx  --shapes=image:16x3x240x240 --threads
```

batch1 t4单卡吞吐率：1000*16/24.3536=656.98fps

### 7.3 性能对比

|               | **310**  | **310P** | **T4** | **310P/310** | **310P/T4** |
| ------------- | -------- | -------- | ------ | ------------ | ----------- |
| **bs1**       | 944.972  | 359.09   | 353.61 | 0.38         | 1.02        |
| **bs16**      | 1361.796 | 1635.3   | 656.98 | 1.201        | 2.49        |
| **最优batch** | 1361.796 | 1635.3   | 656.98 | 1.201        | 2.49        |

batch16： 

310P vs 310: 1635.3fps > 1.2 * 1361.796fps 

310P vs T4 : 1635.3fps > 1.6 * 656.98fps 

性能在310P上的性能达到310的1.2倍,达到T4性能的1.6倍,性能达标。
