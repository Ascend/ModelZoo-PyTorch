

# SPACH 模型端到端推理指导

- [1. 模型概述](#1)
    - [论文地址](#11)
    - [开源代码仓地址](#12)
- [2. 环境说明](#2)
    - [安装依赖](#21)
    - [获取开源模型代码仓](#22)
    - [准备数据集](#23)
- [3. 模型转换](#3)
    - [获取pth文件](#31)
    - [pth转om模型](#32)
- [4. 离线推理](#4)
    - [离线推理](#42)
    - [精度和性能比较](#43)
- [5.性能优化方法](#5)
- [6.性能测试](#6)
- [7.精度和性能评测结果](#7)



## <a name="1">1. 模型概述</a>
### <a name="11">1.1 论文地址</a>
[SPACH论文](https://arxiv.org/abs/2108.13002)

### <a name="12">1.2 开源代码仓地址</a>
[SPACH模型开源代码仓](https://github.com/microsoft/SPACH)

> **说明：**   
> 本离线推理项目中SPACH模型对应开源代码仓中的SPACH-Conv-MS-S

## <a name="2">2. 环境说明</a>
### <a name="21">2.1 安装依赖</a>

```
pip install -r requirements.txt
```
CANN版本
```
CANN 5.1.RC1
```

### <a name="22">2.2 获取开源模型代码仓</a>
```
git clone http://github.com/microsoft/Spach
cd Spach
git reset --hard {commit_id}
cd ..
```

`branch`：main

`{commit_id}`：f69157d4e90fff88285766a4eabf51b29d772da3

### <a name="23">2.3 准备数据集</a>

模型在线推理所用数据集目录结构是torchvision的dataset.ImageFolder标准布局。class_n为数据类别编号。

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

模型离线推理所用数据集为ImageNet的val数据集，310P上数据集所在位置如下

```
/root/datasets/imagenet/val
```

## <a name="3">3. 模型转换</a>

### <a name="31">3.1 获取pth权重文件</a>
获取权重文件 [spach_ms_conv_s.pth](https://github.com/microsoft/SPACH/releases/download/v1.0/spach_ms_conv_s.pth)，并将其放入当前工作目录。

### <a name="32">3.2 pth转om模型</a>

```
bash test/pth2om.sh {chip_name}
```
`{chip_name}`为芯片型号，可通过"npu-smi info"命令查看。在当前目录获得onnx模型和om模型：spach_ms_conv_s.onnx，spach_ms_conv_s_1.om，spach_ms_conv_s_16.om。  

## <a name="4">4. 离线推理</a>

准备 msame 推理工具，将 [msame](https://gitee.com/ascend/tools/tree/master/msame) 安装到当前工作目录，若无权限则需要给msame文件设置权限

```
chmod 777 ./msame
```

一键执行推理

```
bash ./test/eval_acc_perf.sh {val_dataset_path} {val_label_path}
```
`{val_dataset_path}`为用户环境中imagenet的val数集所在目录。

`{val_label_path}`为用户环境中imagenet的val数集标签所在目录。

执行后prep_dataset保存的为预处理数据集结果文件，output为离线推理结果。  

## <a name="5">5. 性能优化方法（命令已嵌入pth2om.sh中，无须单独执行）</a>

对于SPACH模型，直接按照推理文档转换成的模型在性能测试中表现不佳，于是采用该方法对模型性能进行优化。

**atc命令**

```
export TUNE_BANK_PATH=${path_to_custom_tune_bank}/custom_tune_bank/bs${bs}

atc --model=spach_ms_conv_s.onnx --framework=5 --output=spach_bs${bs}\
 --op_precision_mode=op_precision.ini\
 --log=error --input_format=NCHW --input_shape="input:${bs},3,224,224" --
output_type=FP16 --soc_version=${chip_name}
```

变量说明

- `${path_to_custom_tune_bank}` :知识库路径
- `${bs}` : batchsize 大小
- `${chip_name}` :可通过 npu-smi info 命令查看

参数说明

- `model` : 输入的onnx模型路径。

- `output` :输出的文件名。

- `input_format` : 输入形状的格式。

- `input_shape` : 模型输入的形状。

- `op_precision_mode` : 算⼦精度模式配置文件

- `log` : 设置ATC模型转换过程中日志的级别。

- `soc_version` : 芯片型号。

## <a name="6">6.性能测试</a>

```
bash ./test/perf_g.sh {om_model}
```

`{om_model}`为需要进行性能测试的om模型。

## <a name="7">7. 精度和性能评测结果</a>

| 模型            | 参考精度     | 310P 精度   | 基准性能    | 310P 性能   |
| --------------- | ------------ | ----------- | ----------- | ----------- |
| SPACH-Conv-MS-S | acc@1 = 81.6 | acc@1= 81.5 | 276.581 fps | 460.957 fps |

