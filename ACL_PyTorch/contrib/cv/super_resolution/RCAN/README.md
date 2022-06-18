# RCAN Onnx 模型端到端推理指导

- [1. 模型概述](#1)
  - [论文地址](#11)
  - [代码地址](#12)
- [2. 环境说明](#2)
  - [深度学习框架](#21)
  - [python第三方库](#22)
- [3. 模型转换](#3)
  - [pth转onnx模型](#31)
- [4. 数据预处理](#4)
  - [数据集获取](#41)
  - [数据集预处理](#42)
  - [生成数据集信息文件](#43)
- [5. 离线推理](#5)
  - [benchmark工具概述](#51)
  - [离线推理](#52)
- [6. 精度对比](#6)
- [7. 性能对比](#7)
  - [npu性能数据](#71)
  - [T4性能数据](#72)
  - [性能对比](#73)

## <a name="1">1. 模型概述</a>

### <a name="11">1.1 论文地址</a>

[RCAB 论文](https://arxiv.org/abs/1807.02758)

### <a name="12">1.2 代码地址</a>

[RCAN 代码](https://github.com/yulunzhang/RCAN)

branck: master

commit_id: 3339ebc59519c3bb2b5719b87dd36515ec7f3ba7

## <a name="2">2. 环境说明</a>

对于batch1与batch16，310性能均高于T4性能1.2倍，该模型放s在Benchmark/cv/classification目录下。</a>

### <a name="21">2.1 深度学习框架</a>

```
pytorch == 1.8.0
torchvision == 0.9.0
onnx == 1.9.0
CANN == 5.1.RC1
```

### <a name="22">2.2 python第三方库</a>

```
numpy == 1.21.1
Pillow == 7.2.0
opencv-python == 4.2.0
imageio == 2.9.0
scikit-image == 0.18.1
scipy == 1.7.3
```


> **说明：**
>
> X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装 
>
> Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## <a name="3">3. 模型转换</a>

### <a name="31">3.1 pth转onnx模型</a>

1. 下载 pth 权重文，放入models目录下

   [RCAN 预训练pth权重文件](https://pan.baidu.com/s/1bkoJKmdOcvLhOFXHVkFlKA)

   文件名：RCAN_BIX2.pt

   md5sum：f567f8560fde71ba0973a7fe472a42f2

2. 克隆代码仓库代码

   ```bash
   git clone https://github.com/yulunzhang/RCAN.git
   ```

3. 使用rcan_pth2onnx.py 脚本将pth转化为onnx

   ```bash
   python3.7 rcan_pth2onnx.py --pth RCAN_BIX2.pt --onnx rcan.onnx
   ```

   RCAN_BIX2.pt 文件为步骤1中下载的预训练权重文件，该条指令将在运行处生成一个rcan.onnx文件，此文件即为目标onnx文件


### <a name="32">3.2 onnx转om模型</a>

下列需要在具备华为Ascend系列芯片的机器上执行：

1. 设置 atc 工作所需要的环境变量

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
   
2. 由于transpose算子对于某些shape不友好，需要进行优化，将如下内容写入switch.cfg中

   ```
   TransposeReshapeFusionPass:off
   ```

   经过Profiling分析，ConfusionTransposeD算子性能过低，故将其输入加入白名单。即在/usr/local/Ascend/ascend-toolkit/5.0.2.alpha003/x86_64-linux/opp/op_impl/built-in/ai_core/tbe/impl/dynamic/transpose.py里添加 Tranpose shape 白名单：

   ```
   1,64,2,2,256,256
   ```

   以下是优化前后性能对比

   |      | 未任何优化前310（单卡吞吐率） | 优化后310（单卡吞吐率） |
   | :--: | :---------------------------: | :---------------------: |
   | bs1  |            0.7245             |         9.3220          |

3. 使用atc工具将onnx模型转换为om模型，命令参考

   ${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)
   
   ```bash
   atc --framework=5 --model=rcan.onnx --output=rcan_1bs --input_format=NCHW --input_shape="image:1,3,256,256" --fusion_switch_file=switch.cfg --log=debug --soc_version=Ascend${chip_name}
   ```

   此命令将在运行路径下生成一个rcan_1bs.om文件，此文件即为目标om模型文件

## <a name="4">4. 数据预处理</a>

### <a name="41">4.1 数据集获取</a>

该模型使用[Set5](https://github.com/yulunzhang/RCAN/tree/master/RCAN_TestCode/OriginalTestData/Set5)的5张验证集进行测试，图片数据放在/root/datasets/Set5。

### <a name="42">4.2 数据集预处理</a>

使用 rcan_preprocess.py 脚本进行数据预处理，脚本执行命令：

```
python3.7 rcan_preprocess.py -s /root/datasets/Set5/LR -d ./prep_data --size 256
```

由于rcan模型支持动态输入，而atc工具需要指定输入大小，所以要在此对图像添加pad和进行缩放到同一大小，最终对推理产生的结果进行后处理恢复。以上命令将自动生成一个pad_info.json文件，此文件记录在数据预处理中对图像的pad和缩放信息，用于数据后处理时进行图像裁剪。

### <a name="43">4.3 生成数据集信息文件</a>

1. 生成数据集信息文件脚本 gen_dataset_info.py

2. 执行生成数据集信息脚本，生成数据集信息文件

   ```bash
   python3.7 gen_dataset_info.py bin ./prep_data ./rcan_prep_bin.info 256 256
   ```

   第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息

## <a name="5">5. 离线推理</a>

### <a name="51">5.1 benchmark工具概述</a>

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN V100R020C10 推理benchmark工具用户指南 01

### <a name="52">5.2 离线推理</a>

1. 执行离线推理

   ```bash
   ./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=rcan_1bs.om -input_text_path=./rcan_prep_bin.info -input_width=256 -input_height=256 -output_binary=True -useDvpp=False
   ```

   输出结果默认保存在当前目录result/dumpOutput_device{0}，模型只有一个名为HR_image的输出，shape为bs * 512 * 512，数据类型为FP32，对应1个超分辨后的图像数据，每个输入对应的输出对应一个_x.bin文件。
   
2. 数据后处理

   ```bash
   python3.7 rcan_postprocess.py -s result/dumpOutput_device0/ -d post_data
   ```

   由于在预处理中对图像进行了添加pad和缩放操作，故要对推理结果进行相应的裁剪和缩放

## <a name="6">6. 精度对比</a>

### <a name="61">6.1 离线推理TopN精度</a>

## <a name="6">6. 精度对比</a>

|                    | PSNR  |  SSIM  |
| :----------------: | :---: | :----: |
|   原github仓库精度   | 38.27 | 0.9614 |
| 310 om模型离线推理精度 | 38.25 | 0.9606 |
| 310p om模型离线推理精度 | 38.25 | 0.9606 |

## <a name="7">7. 性能对比</a>

### <a name="71">7.1 npu性能数据</a>

benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。由于模型不接受多 batch 输入，所以只统计在整个数据集上推理得到bs1的性能数据

使用benchmark工具在整个数据集上推理时获得的性能数据：

310:
```
[e2e] throughputRate: 1.30679, latency: 3826.82
[data read] throughputRate: 1058.2, moduleLatency: 0.945
[preprocess] throughputRate: 7.64831, moduleLatency: 130.748
[infer] throughputRate: 2.3029, Interface throughputRate: 2.32143, moduleLatency: 433.593
[post] throughputRate: 2.75189, moduleLatency: 363.387
```

Interface throughputRate: 2.32143，2.32143x4=9.28572fps 即是batch1 310单卡吞吐率

310p:
```
[e2e] throughputRate: 2.72746, latency: 1833.21
[data read] throughputRate: 1405.68, moduleLatency: 0.7114
[preprocess] throughputRate: 8.26564, moduleLatency: 120.983
[infer] throughputRate: 11.5912, Interface throughputRate: 12.4141, moduleLatency: 86.0648
[post] throughputRate: 13.6097, moduleLatency: 73.4772
```

Interface throughputRate: 12.4141, 即是batch1 310p单卡吞吐率


### <a name="72">7.2 T4性能数据</a>

在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2

```
trtexec --onnx=rcan.onnx --fp16 --shapes=image:1x3x256x256
```

gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch。其中--fp16是算子精度，目前算子精度只测--fp16的。

```
[07/14/2021-10:48:26] [I] GPU Compute
[07/14/2021-10:48:26] [I] min: 150.972 ms
[07/14/2021-10:48:26] [I] max: 157.659 ms
[07/14/2021-10:48:26] [I] mean: 152.567 ms
[07/14/2021-10:48:26] [I] median: 151.477 ms
[07/14/2021-10:48:26] [I] percentile: 157.659 ms at 99%
[07/14/2021-10:48:26] [I] total compute time: 3.19929 s
```

batch1 t4单卡吞吐率：1000/(152.567/1)=6.5545fps 

### <a name="73">7.3 性能对比</a>
|       | 310   |  310p  |   T4   | 310p/310 | 310p/T4|
| :---: | :---: | :----: | :----: |  :---:   | :----: |
|  bs1  |9.28572|12.4141 | 6.5545 |  1.3369  | 1.8940 |

310单个device的吞吐率乘4即单卡吞吐率比T4单卡的吞吐率大，故310性能高于T4性能，性能达标。
310p单卡的吞吐率大于310单卡吞吐率乘4的1.2倍，310p单卡大于T4单卡的1.6倍，性能达标。

