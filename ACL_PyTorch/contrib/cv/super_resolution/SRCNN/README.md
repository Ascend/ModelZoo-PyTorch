# SRCNN Onnx 模型端到端推理指导

- [1. 模型概述](#1)
  - [论文地址](#11)
  - [代码地址](#12)
- [2. 环境说明](#2)
  - [深度学习框架](#21)
  - [python第三方库](#22)
- [3. 模型转换](#3)
  - [pth转onnx模型](#31)
  - [onnx转om模型](#32)
- [4. 数据预处理](#4)
  - [数据集获取](#41)
  - [数据集预处理](#42)
  - [生成数据集信息文件](#43)
- [5. 离线推理](#5)
  - [benchmark工具概述](#51)
  - [离线推理](#52)
- [6. 精度对比](#6)
  - [离线推理精度](#61)
  - [精度对比](#62)
- [7. 性能对比](#7)
  - [npu性能数据](#71)
  - [T4性能数据](#72)
  - [性能对比](#73)

## <a name="1">1. 模型概述</a>

### <a name="11">1.1 论文地址</a>

[SRCNN 论文](https://arxiv.org/abs/1501.00092) 

### <a name="12">1.2 代码地址</a>

[SRCNN 代码](https://github.com/yjn870/SRCNN-pytorch)

branch: master

commit_id: 064dbaac09859f5fa1b35608ab90145e2d60828b

## <a name="2">2. 环境说明</a>

### <a name="21">2.1 深度学习框架</a>

```
pytorch == 1.5.0
torchvision == 0.6.0
onnx == 1.9.0
```

### <a name="22">2.2 python第三方库</a>

```
numpy == 1.19.2
Pillow == 8.2.0
opencv-python == 4.5.2
```

> **说明：**
>
> X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装 
>
> Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## <a name="3">3. 模型转换</a>

### <a name="31">3.1 pth转onnx模型</a>

1. 下载 pth 权重文件

   [SRCNN_x2预训练pth权重文件](https://www.dropbox.com/s/rxluu1y8ptjm4rn/srcnn_x2.pth?dl=0)

   文件名：srcnn_x2.pth

   md5sum：e0a9e64cf1f9016d7013e0b01f613f68

2. 克隆代码仓库代码

   ```bash
   git clone https://github.com/yjn870/SRCNN-pytorch
   ```

3. 使用 srcnn_pth2onnx.py 转换pth为onnx文件，在命令行运行如下指令：

   ```bash
   python3.7 srcnn_pth2onnx.py --pth srcnn_x2.pth --onnx srcnn_x2.onnx
   ```
   
   srcnn_x2.pth文件为步骤1中下载的预训练权重文件，该条指令将在运行处生成一个srcnn_x2.onnx文件，此文件即为目标onnx文件

### <a name="32">3.2 onnx转om模型</a>

下列需要在具备华为Ascend系列芯片的机器上执行：

1. 设置 atc 工作所需要的环境变量

   ```bash
   export install_path=/usr/local/Ascend/ascend-toolkit/latest
   export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
   export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
   export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
   export ASCEND_OPP_PATH=${install_path}/opp
   ```

2. 使用atc工具将onnx模型转换为om模型，命令参考

   ```bash
    atc --framework=5 --model=srcnn_x2.onnx --output=srcnn_x2 --input_format=NCHW --input_shape="input.1:1,1,256,256" --log=debug --soc_version=Ascend310
   ```

   此命令将在运行路径下生成一个srcnn_x2.om文件，此文件即为目标om模型文件

## <a name="4">4. 数据预处理</a>

### <a name="41">4.1 数据集获取</a>

该模型使用[Set5官网](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)的5张验证集进行测试，图片存放在/root/dataset/set5下面。

### <a name="42">4.2 数据集预处理</a>

使用 srcnn_preprocess.py 脚本进行数据预处理，脚本执行命令：

```bash
python3.7 srcnn_preprocess.py -s /root/datasets/set5 -d ./prep_data
```

预处理脚本会在./prep_data/png/下保存中心裁剪大小为256*256的预处理图片，并在缩放处理之后将bin文件保存至./prep_data/bin/下面。

### <a name="43">4.3 生成数据集信息文件</a>

1. 生成数据集信息文件脚本 get_info.py

2. 执行生成数据集信息脚本，生成数据集信息文件

   ```bash
    python3.7 get_info.py bin ./prep_data/bin ./srcnn_prep_bin.info 256 256
   ```

   第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息

## <a name="5">5. 离线推理</a>

### <a name="51">5.1 benchmark工具概述</a>

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN V100R020C10 推理benchmark工具用户指南 01

### <a name="52">5.2 离线推理</a>

```bash
./benchmark -model_type=vision -device_id=0 -batch_size=1 -om_path=./srcnn_x2.om -input_text_path=./srcnn_prep_bin.info -input_width=256 -input_height=256 -output_binary=False -useDvpp=False
```

输出结果默认保存在当前目录result/dumpOutput_device{0}，对应了Set5中每张图片的输出。

## <a name="6">6. 精度对比</a>

### <a name="61">6.1 离线推理精度</a>

后处理输出每一张图片在经过SRCNN处理之后的PSRN的值，同时将处理后的图片保存在./result/save目录下。调用srcnn_postprocess.py来进行后处理，结果输出在控制台上。

```bash
python3.7 srcnn_postprocess.py --res ./result/dumpOutput_device0/ --png_src ./prep_data/png --bin_src ./prep_data/bin --save ./result/save
```

第一行为处理的图片名，第二行为PSNR的计算结果：

```bash
woman_256.png PSNR: 36.79

bird_256.png PSNR: 37.61

head_256.png PSNR: 41.38

baby_256.png PSNR: 36.31

butterfly_256.png PSNR: 29.59

total PSNR: 36.33

```
上述结果中的total psnr是通过对于所有psnr结果取平均值得到的。

### <a name="62">6.2 精度对比</a>

根据该模型github上代码仓所提供的测试脚本，运行pth文件，得到预训练文件在官方代码仓库上对于Set5的验证集的PSNR输出值。

```bash
cd SRCNN-pytorch
echo "woman "
python3.7 test.py --weights-file "../srcnn_x2.pth" \
               --image-file "../prep_data/png/woman_256.png" \
               --scale 2
echo ""
echo "bird "
python3.7 test.py --weights-file "../srcnn_x2.pth" \
               --image-file "../prep_data/png/bird_256.png" \
               --scale 2
echo ""
echo "head "
python3.7 test.py --weights-file "../srcnn_x2.pth" \
               --image-file "../prep_data/png/head_256.png" \
               --scale 2
echo ""
echo "baby "
python3.7 test.py --weights-file "../srcnn_x2.pth" \
               --image-file "../prep_data/png/baby_256.png" \
               --scale 2
echo ""
echo "butterfly "
python3.7 test.py --weights-file "../srcnn_x2.pth" \
               --image-file "../prep_data/png/butterfly_256.png" \
               --scale 2
echo ""
```
可以得到：

|         |   原github测试脚本PSNR   |   om模型PSNR   |
| :-----: | :------: | :------: |
|  woman  | 36.89 | 36.79  |
| bird |  37.74  |  37.61  |
| head |  41.24  |  41.38  |
| baby |  36.52  |  36.31  |
| butterfly |  29.55  |  29.59  |

将得到的om离线模型推理精度与该模型github代码仓上公布的精度对比，精度下降均在1%范围之内。同时，total psnr为36.33dB，对比开源仓库的36.65dB而言，下降精度0.9%也在1%以内，故精度达标。

## <a name="7">7. 性能对比</a>

### <a name="71">7.1 npu性能数据</a>

benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。

benchmark工具作纯推理时使用的命令参考如下：

```bash
./benchmark -round=20 -om_path=./srcnn_x2.om -device_id=0 -batch_size=1
```

benchmark工具在整个数据集上的运行结果如下：
   ```
    [e2e] throughputRate: 3.74691, latency: 1334.43
    [data read] throughputRate: 2857.14, moduleLatency: 0.35
    [preprocess] throughputRate: 312.48, moduleLatency: 3.2002
    [infer] throughputRate: 285.682, Interface throughputRate: 348.038, moduleLatency: 3.4136
    [post] throughputRate: 50.2841, moduleLatency: 19.887
   ```

   Interface throughputRate: 348.038 * 4 = 1392.152 即是batch1 310单卡吞吐率

### <a name="72">7.2 T4性能数据</a>

在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2

```bash
trtexec --onnx=srcnn_x2.onnx --fp16 --shapes=image:1x1x256x256 --threads
```

gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch。其中--fp16是算子精度，目前算子精度只测--fp16的。

   ```
    [07/18/2021-15:30:00] [I] GPU Compute
    [07/18/2021-15:30:00] [I] min: 0.796875 ms
    [07/18/2021-15:30:00] [I] max: 3.98999 ms
    [07/18/2021-15:30:00] [I] mean: 0.859765 ms
    [07/18/2021-15:30:00] [I] median: 0.853516 ms
    [07/18/2021-15:30:00] [I] percentile: 0.987305 ms at 99%
    [07/18/2021-15:30:00] [I] total compute time: 2.98597 s
   ```
   
   batch1 t4单卡吞吐率：1000/(0.859765/1)=1163.108fps
   
### <a name="73">7.3 性能对比</a>

batch1：1392.152fps > 1163.108fps

310单个device的吞吐率乘4即单卡吞吐率比T4单卡的吞吐率大，故310性能高于T4性能，性能达标。
