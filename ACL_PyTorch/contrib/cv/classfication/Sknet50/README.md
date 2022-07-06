# SK-ResNet50 Onnx 模型端到端推理指导 

- [1. 模型概述](#1)
  - [论文地址](#11)
  - [代码地址](#12)
- [2. 环境说明](#2)
  - [深度学习框架](#21)
  - [python第三方库](#22)
  - [环境导入命令](#23)
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
  - [离线推理TopN精度](#61)
  - [精度对比](#62)
- [7. 性能对比](#7)
  - [npu性能数据](#71)
  - [T4性能数据](#72)
  - [性能对比](#73)

## <a name="1">1. 模型概述</a>

### <a name="11">1.1 论文地址</a>

[SK-ResNet 论文](https://arxiv.org/pdf/1903.06586.pdf) 

### <a name="12">1.2 代码地址</a>

[SK-ResNet 代码](https://github.com/implus/PytorchInsight)

branch: master

commit_id: 2864528f8b83f52c3df76f7c3804aa468b91e5cf

## <a name="2">2. 环境说明</a>

### <a name="21">2.1 深度学习框架</a>

```
pytorch == 1.8.2+cpu
torchvision == 0.9.2+cpu
onnx == 1.9.0
```

### <a name="22">2.2 python第三方库</a>

```
numpy == 1.21.6
Pillow == 9.1.0
opencv-python == 4.5.5.64
sympy == 1.10.1
decorator == 5.1.1

```

### <a name="23">2.3 环境导入命令</a>

```
pip install sympy
pip install decorator
pip install onnx==1.9.0
pip install torch==1.8.2+cpu torchvision==0.9.2+cpu torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install opencv-python
```

> **说明：**
>
> X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装 
>
> Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## <a name="3">3. 模型转换</a>

### <a name="31">3.1 pth转onnx模型</a>

1. 下载 pth 权重文件

   [SK-ResNet50预训练pth权重文件(百度网盘，提取码：tfwn)](https://pan.baidu.com/s/1Lx5CNUeRQXOSWjzTlcO2HQ)

   文件名：sk_resnet50.pth.tar

   md5sum：979bbb525ee0898003777a8e663e91c0

2. 克隆代码仓库代码

   ```bash
   git clone https://github.com/implus/PytorchInsight.git
   ```
	仓库代码克隆到Sknet50文件夹下
	仓库代码包含pth权重文件

3. 使用 sknet2onnx.py 转换pth为onnx文件，在命令行运行如下指令：

   ```bash
   python sknet2onnx.py --pth sk_resnet50.pth.tar --onnx sk_resnet50.onnx
   ```
   
   sk_resnet50.pth.tar文件为步骤1中下载的预训练权重文件，该条指令将在运行处生成一个sknet50.onnx文件，此文件即为目标onnx文件

**模型转换要点：**

> pytorch导出onnx时softmax引入了transpose以操作任意轴，然而在onnx中已支持softmax操作任意轴，故可删除transpose提升性能

### <a name="32">3.2 onnx转om模型</a>

下列需要在具备华为Ascend系列芯片的机器上执行：

1. 设置 atc 工作所需要的环境变量

   ```bash
   source set_env.sh
   ```

2. 使用atc工具将onnx模型转换为om模型，命令参考

   ${chip_name}可通过`npu-smi info`指令查看

   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

   ```bash
   1batch命令:
   atc --framework=5 --model=sk_resnet50.onnx --output=sk_resnet50_bs1_310p --input_format=NCHW --input_shape="image:1,3,224,224" --log=debug --soc_version=Ascend${chip_name}
   ```
   
   1batch命令将在运行路径下生成一个Sk_resnet50_bs1_310p.om文件，此文件即为1batch的om模型文件，其他batch同理

## <a name="4">4. 数据预处理</a>

### <a name="41">4.1 数据集获取</a>

该模型使用[ImageNet官网](http://www.image-net.org/)的5万张验证集进行测试，图片与标签分别存放在/opt/npu/imageNet/val与/opt/npu/imageNet/val_label_LSL.txt。

### <a name="42">4.2 数据集预处理</a>

使用 sknet_preprocess.py 脚本进行数据预处理，脚本执行命令：

```bash
python sknet_preprocess.py -s /opt/npu/imageNet/val -d ./prep_data
```

### <a name="43">4.3 生成数据集信息文件</a>

1. 生成数据集信息文件脚本 get_info.py

2. 执行生成数据集信息脚本，生成数据集信息文件

   ```bash
   python get_info.py bin ./prep_data ./sknet_prep_bin_lsl.info 224 224
   ```

   第一个参数为模型输入的类型，第二个参数为生成的bin文件路径，第三个为输出的info文件，后面为宽高信息

## <a name="5">5. 离线推理</a>

### <a name="51">5.1 benchmark工具概述</a>

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310p上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考CANN V100R020C10 推理benchmark工具用户指南 01

### <a name="52">5.2 离线推理</a>

```bash
chmod +x benchmark.x86_64
1batch:
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=sk_resnet50_bs1_310p.om -input_text_path=sknet_prep_bin_lsl.info -input_width=224 -input_height=224 -output_binary=False -useDvpp=False

```
首先要添加benchmark.x86_64可执行权限再进行推理
输出结果默认保存在当前目录result/dumpOutput_device{0}，模型只有一个名为class的输出，shape为bs * 1000，数据类型为FP32，对应1000个分类的预测结果，每个输入对应的输出对应一个_x.bin文件。上面为1batch，其他batch同理

## <a name="6">6. 精度对比</a>

### <a name="61">6.1 离线推理TopN精度</a>

后处理统计TopN精度，调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中：

```bash
python vision_metric_ImageNet.py result/dumpOutput_device0/ /opt/npu/imageNet/val_label_LSL.txt ./ result_bs1.json
```

第一个为benchmark输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名。查看输出结果：

```json
{"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "77.54%"}, {"key": "Top2 accuracy", "value": "87.12%"}, {"key": "Top3 accuracy", "value": "90.73%"}, {"key": "Top4 accuracy", "value": "92.55%"}, {"key": "Top5 accuracy", "value": "93.70%"}]}
```

经过对bs1与bs16的om测试，本模型batch1的精度与batch16的精度没有差别，精度数据均如上

### <a name="62">6.2 精度对比</a>

|                    |   TOP1   |   TOP5   |
| :----------------: | :------: | :------: |
|  310精度  | 77.54% | 93.70% |
|  310p精度  | 77.54% | 93.70% |

将得到的om离线模型推理TopN精度与该模型github代码仓上公布的精度对比，精度下降在1%范围之内，故精度达标。

## <a name="7">7. 性能对比</a>

### <a name="71">7.1 npu性能数据</a>

benchmark工具在整个数据集上推理时也会统计性能数据，但是推理整个数据集较慢，如果这么测性能那么整个推理期间需要确保独占device，使用npu-smi info可以查看device是否空闲。也可以使用benchmark纯推理功能测得性能数据，但是由于随机数不能模拟数据分布，纯推理功能测的有些模型性能数据可能不太准，benchmark纯推理功能测性能仅为快速获取大概的性能数据以便调试优化使用，可初步确认benchmark工具在整个数据集上推理时由于device也被其它推理任务使用了导致的性能不准的问题。模型的性能以使用benchmark工具在整个数据集上推理得到bs1与bs16的性能数据为准，对于使用benchmark工具测试的batch4，8，32的性能数据在README.md中如下作记录即可。

benchmark工具作纯推理时使用的命令参考如下：

```bash
./benchmark.x86_64 -round=20 -om_path=sknet50_1bs.om -batch_size=1 
```

1. batch1 性能

   使用benchmark工具在整个数据集上推理时获得的性能数据：

   Interface throughputRate: 812.681 即是batch1 310p单卡吞吐率

2. batch4 性能

   Interface throughputRate: 2062.16 即是batch4 310p单卡吞吐率

3. batch8 性能

   Interface throughputRate: 2174.18 即是batch8 310p单卡吞吐率

4. batch16 性能

   Interface throughputRate: 2041.62 即是batch16 310p单卡吞吐率

5. batch32 性能

   Interface throughputRate: 1879.74 即是batch32 310p单卡吞吐率

6. batch64 性能

   Interface throughputRate: 1780.68 即是batch64 310p单卡吞吐率

### <a name="72">7.2 T4性能数据</a>

在装有T4卡的服务器上测试gpu性能，测试过程请确保卡没有运行其他任务，TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2

```bash
trtexec --onnx=sk_resnet50.onnx --fp16 --shapes=image:1x3x224x224 --threads
```

gpu T4是4个device并行执行的结果，mean是时延（tensorrt的时延是batch个数据的推理时间），即吞吐率的倒数乘以batch。其中--fp16是算子精度，目前算子精度只测--fp16的。

1. batch1 性能

   batch1 t4单卡吞吐率：1000/(2.16149/1)=462.64fps
   
2. batch4 性能

   batch4 t4单卡吞吐率：1000/(4.43787/4)=895.11fps
   
3. batch8 性能
 
   batch8 t4单卡吞吐率：1000/(7.84424/8)=1019.86fps
   
4. batch16 性能
 
   batch16 t4单卡吞吐率：1000/(15.3644/16)=1041.37fps
   
5. batch32 性能
 
   batch32 t4单卡吞吐率：1000/(26.6422/32)=1201.10fps

6. batch64 性能
 
   batch64 t4单卡吞吐率：1000/(50.5169/64)=1266.90fps

### <a name="73">7.3 性能对比</a>

|       |    310   |    310p   |     T4   | 310p/310  |  310p/T4  |
| :-----| :------: | :------: | :------: | :------: | :------: |
|  bs1  | 723.5 | 812.681 | 462.64 | 1.123263303 | 1.756616376 |
|  bs4  | 1177.36 | 2062.16 | 895.11 | 1.751511857 | 2.303806236 |
|  bs8  | 1325.04 | 2174.18 | 1019.86 | 1.640841031 | 2.131841625 |
|  bs16  | 1357.24 | 2041.62 | 1041.37 | 1.504243907 | 1.960513554 |
|  bs32  | 1329.876 | 1879.74 | 1201.1 | 1.413470128 | 1.565015403 |
|  bs64  | 1165.084 | 1780.68 | 1266.9 | 1.528370487 | 1.405541085 |
|  最优bs | 1357.24 | 2174.18 | 1266.9 | 1.601912705 | 1.716141763 |

310P的每个batch的性能都需要达到310的1倍以上，性能达标。