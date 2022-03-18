# EDSR Onnx 模型端到端推理指导

- [1. 模型概述](#1)
  - [论文地址](#11)
  - [代码地址](#12)
- [2. 环境说明](#2)
  - [深度学习框架](#21)
  - [python第三方库](#22)
- [3. 模型转换](#3)
  - [pth转onnx模型](#31)
  - [onnx转om模型](#32)
  - [om模型性能提升](#33)
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

[EDSR 论文](https://arxiv.org/abs/1707.02921) 

### <a name="12">1.2 代码地址</a>

[EDSR 代码](https://github.com/sanghyun-son/EDSR-PyTorch)

branch: master

commit_id: 9d3bb0ec620ea2ac1b5e5e7a32b0133fbba66fd2

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

   [EDSR_x2预训练pth权重文件](https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt)

   文件名：edsr_baseline_x2-1bc95232.pt

   md5sum：e0a9e64cf1f9016d7013e0b01f613f68

2. 克隆代码仓库代码

   ```bash
   git clone https://github.com/sanghyun-son/EDSR-PyTorch
   ```
   EDSR的github仓库重写了Model的forward方法，并给forward添加了一个idx_scale的参数，这会使得PyTorch官方转换onnx的API失效，运行如下命令，将onnx.diff打包到原仓库中，使得在不影响原仓库功能的前提下实现对官方转换api的支持。

   ```bash
   patch -p1 < ../edsr.diff
   ```

3. 确定onnx输入输出的尺寸
   为了增加精度，本指导采用对于不满足尺寸大小要求的图像的右侧和下方填充0的方式来使其输入图像达到尺寸大小要求。因此首先要获得需要的尺寸大小，通过命令行中运行如下脚本：

   ```bash
   python3.7 get_max_size.py --dir /root/datasets/div2k/LR
   ```

   对于div2k数据集中scale为2的缩放，尺寸大小应为1020。

4. 使用 edsr_pth2onnx.py 转换pth为onnx文件，在命令行运行如下指令：

   ```bash
   python3.7 edsr_pth2onnx.py --pth edsr_x2.pt --onnx edsr_x2.onnx --size 1020
   ```
   
   edsr_x2.pt文件为步骤1中下载的预训练权重文件，该条指令将在运行处生成一个edsr_x2.onnx文件，此文件即为目标onnx文件

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
   atc --framework=5 --model=edsr_x2.onnx --output=edsr_x2 --input_format=NCHW --input_shape="input.1:1,3,1020,1020" --log=debug --soc_version=Ascend310
   ```

   此命令将在运行路径下生成一个edsr_x2.om文件，此文件即为目标om模型文件

### <a name="33">3.3 om模型性能提升</a>
   直接使用atc工具转换会将Transpose算子翻译为TransposeD，而TransposeD在一些输入形状下会有很差的性能。因此首先需要将Transpose的输入形状加入atc转换的白名单中。

1. 获取onnx模型各个节点的输入输出shape

   onnx模型默认并不带有输入输出的shape信息，可以使用 infer_onnx_shape.py将输入输出的shape信息添加到onnx中。

   ```bash
   python3.7 infer_onnx_shape.py --onnx edsr_x2.onnx
   ```

   运行之后从服务器上下载edsr_x2.onnx，并用netron打开，可以得到Transpose算子的输入为 [1, 64, 2, 2, 1020, 1020]。
2. 将输入shape添加到白名单中

   打开 /usr/local/Ascend/ascend-toolkit/5.0.2.alpha005/x86_64-linux/opp/op_impl/built-in/ai_core/tbe/impl/dynamic/transpose.py 文件，在函数 _by_dynamic_static_union_version 的 white_list_shape 列表中加入Transpose的输入shape。

   ```python
   def _by_dynamic_static_union_version(shape, core_num):
   ...
      white_list_shape = [
         ...
         [1, 64, 2, 2, 1020, 1020]
      ]
   ```
3. 关闭TransposeReshapeFusionPass

   在atc转换命令中加入 --fusion_switch_file=switch.cfg，关闭TransposeReshapeFusionPass。

   ```bash
   atc --framework=5 --model=edsr_x2.onnx --output=edsr_x2 --input_format=NCHW --input_shape="input.1:1,3,1020,1020" --log=debug --soc_version=Ascend310 --fusion_switch_file=switch.cfg
   ```

## <a name="4">4. 数据预处理</a>

### <a name="41">4.1 数据集获取</a>

该模型使用[DIV2K官网](https://data.vision.ee.ethz.ch/cvl/DIV2K/)的100张验证集进行测试，图片存放在/root/dataset/div2k下面。

其中，低分辨率图像(LR)采用bicubic x2处理(Validation Data Track 1 bicubic downscaling x2 (LR images))，高分辨率图像(HR)采用原图验证集(Validation Data (HR images))。

### <a name="42">4.2 数据集预处理</a>

使用 edsr_preprocess.py 脚本进行数据预处理，脚本执行命令：

```bash
python3.7 edsr_preprocess.py -s /root/datasets/div2k/LR -d ./prep_data --save_img
```

预处理脚本会在./prep_data/png/下保存填充为1020x1020的预处理图片，并将bin文件保存至./prep_data/bin/下面。

### <a name="43">4.3 生成数据集信息文件</a>

1. 生成数据集信息文件脚本 get_info.py

2. 执行生成数据集信息脚本，生成数据集信息文件

   ```bash
    python3.7 get_info.py bin ./prep_data/bin ./edsr_prep_bin.info 1020 1020
   ```

   第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息

## <a name="5">5. 离线推理</a>

### <a name="51">5.1 benchmark工具概述</a>

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN V100R020C10 推理benchmark工具用户指南 01

### <a name="52">5.2 离线推理</a>

```bash
./benchmark -model_type=vision -device_id=0 -batch_size=1 -om_path=edsr_x2.om -input_text_path=./edsr_prep_bin.info -input_width=1020 -input_height=1020 -output_binary=True -useDvpp=False
```

输出结果默认保存在当前目录result/dumpOutput_device{0}，对应了DIV2K中每张图片的超分输出结果。

## <a name="6">6. 精度对比</a>

### <a name="61">6.1 离线推理精度</a>

后处理输出每一张图片在经过EDSR处理之后的PSRN的值，同时将处理后的图片保存在./result/save目录下。调用edsr_postprocess.py来进行后处理，结果输出在控制台上。

```bash
python3.7 edsr_postprocess.py --res ./result/dumpOutput_device0/ --HR /root/datasets/div2k/HR
```

精度计算结果保存在result.json里面

```json
{
    "accuracy": 34.606869807078574,
    "data": [
       ...
       ]
}
```

### <a name="62">6.2 精度对比</a>

github仓库中给出的官方精度为34.61dB，npu离线推理的精度为34.60dB。

将得到的om离线模型推理精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。

## <a name="7">7. 性能对比</a>

### <a name="71">7.1 npu性能数据</a>

由于T4服务器上的显卡显存有限，对于输入为1x3x1020x1020的onnx模型没法正常得到推理结果。因此本指导选择size为256的onnx与om模型作为性能比较的参考对象。通过如下命令获得对应尺寸的onnx和om模型。

```bash
python3.7 edsr_pth2onnx.py --pth edsr_x2.pt --onnx edsr_x2_256.onnx --size 256
```

参照3.3所描述的方法，将Transpose的输入[1, 64, 2, 2, 256, 256]加入优化白名单。随后运行

```bash
source env.sh
atc --framework=5 --model=edsr_x2_256.onnx --output=edsr_x2_256 --input_format=NCHW --input_shape="input.1:1,3,256,256" --log=debug --soc_version=Ascend310 --fusion_switch_file=switch.cfg
```
得到size为256的om模型。

benchmark工具作纯推理测试性能使用的命令参考如下：

```bash
./benchmark -round=20 -om_path=./edsr_x2_256.om -device_id=0 -batch_size=1
```

纯推理的运行结果如下：

   ave_throughputRate = 22.3706samples/s, ave_latency = 44.7073ms

   Interface throughputRate: 22.3706 * 4 = 89.4824 即是batch1 310单卡吞吐率

### <a name="72">7.2 T4性能数据</a>

在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2

```bash
trtexec --onnx=edsr_x2_256.onnx --fp16 --shapes=image:1x3x256x256 --threads
```

gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch。其中--fp16是算子精度，目前算子精度只测--fp16的。

   ```
   [07/28/2021-06:56:34] [I] GPU Compute
   [07/28/2021-06:56:34] [I] min: 11.1594 ms
   [07/28/2021-06:56:34] [I] max: 16.341 ms
   [07/28/2021-06:56:34] [I] mean: 11.7199 ms
   [07/28/2021-06:56:34] [I] median: 11.6487 ms
   [07/28/2021-06:56:34] [I] percentile: 15.5259 ms at 99%
   [07/28/2021-06:56:34] [I] total compute time: 3.02374 s
   ```
   
   batch1 t4单卡吞吐率：1000/(11.6487/1)=85.8465fps
   
### <a name="73">7.3 性能对比</a>

batch1：89.4824fps > 85.8465fps

310单个device的吞吐率乘4即单卡吞吐率比T4单卡的吞吐率大，故310性能高于T4性能，性能达标。
