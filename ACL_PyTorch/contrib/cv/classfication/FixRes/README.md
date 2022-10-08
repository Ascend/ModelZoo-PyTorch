# FixRes Onnx模型端到端推理指导

## 1 模型概述

FixRes是图像分类任务的卷积神经网络，该网络基于ResNet50进行了改进，相比ResNet网络，FixRes在测试时采用更大的分辨率输入图像，以此降低训练、测试时图像增强方法不同对分类准确率造成的负面影响。

### 1.1 论文地址

[Hugo Touvron and Andrea Vedaldi and Matthijs Douze and Hervé Jégou (2020). Fixing the train-test resolution discrepancy: FixEfficientNet. CoRR, abs/2003.08237.](https://arxiv.org/pdf/2003.08237.pdf)

### 1.2 代码地址

```shell
ur=https://github.com/facebookresearch/FixRes
branch=master
commit_id=c9be6acc7a6b32f896e62c28a97c20c2348327d3
```

## 2 环境准备 

### 2.1 深度学习框架

```
CANN 5.1.RC1
pytorch == 1.8.0
torchvision == 0.9.0
onnx == 1.8.0
```

### 2.2 python第三方库

```
numpy == 1.18.5
opencv-python == 4.5.2.54
Pillow == 7.2.0
```

**说明：** 

>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

### 3.1 pth转onnx模型

1. 准备pth权重文件  
   使用训练好的pkl权重文件：ResNetFinetune.pth

下载地址： [https://dl.fbaipublicfiles.com/FixRes_data/FixRes_Pretrained_Models/ResNetFinetune.pth](https://github.com/facebookresearch/FixRes)

2. 导出onnx文件。

   1. 使用“ResNetFinetune.pth”导出onnx文件。

      运行“FixRes_pth2onnx.py”脚本，获得“FixRes.onnx”文件。

      ```shell
      python3.7 FixRes_pth2onnx.py --pretrain_path ResNetFinetune.pth
      ```

      使用ATC工具将.onnx文件转换为.om文件，导出.onnx模型文件时需设置算子版本为11。

### 3.2 onnx模型转om模型

使用ATC工具将ONNX模型转OM模型。

1. 配置环境变量。

   ```shell
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

2. 使用atc将onnx模型
   ${chip_name}可通过npu-smi info指令查看，例：310P3
   ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

执行ATC命令：

```shell
atc --framework=5 
--model=FixRes.onnx 
--output=FixRes_bs1 
--input_format=NCHW 
--input_shape="image:1,3,384,384" 
--log=debug 
--soc_version=Ascend${chip_name}
--auto_tune_mode="RL,GA"
```

参数说明：
--model：为ONNX模型文件。

--framework：5代表ONNX模型。

--output：输出的OM模型。

--input_format：输入数据的格式。

--input_shape：输入数据的shape。

--log：日志级别。

--soc_version：处理器型号。

## 4 数据集预处理

### 4.1 数据集获取

本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到ILSVRC2012_img_val.tar验证集，请自行下载验证需要的标签文件“imagenet_labels_fixres.json”。

数据目录结构请参考：

```
├──ImageNet
    ├──ILSVRC2012_img_val
    ├──imagenet_labels_fixres.json
```

### 4.2 数据预处理。

数据预处理将原始数据集转换为模型输入的数据。

执行“FixRes_preprocess.py”脚本，完成预处理。

```shell
python3.7 FixRes_preprocess.py 
--src-path /home/HwHiAiUser/dataset/imagenet/val 
--save-path ./val_FixRes
```

--src-path：原始数据验证集（.jpeg）所在路径。

--save-path：输出的二进制文件（.bin）所在路径。

每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“val_FixRes”二进制文件夹。

### 4.3 生成数据集info文件。

生成bin文件的输入info文件。

使用benchmark推理需要输入图片数据集的info文件，用于获取数据集。使用“gen_dataset_info.py”脚本，输入已经获得的图片文件，输出生成图片数据集的info文件。运行“gen_dataset_info.py”脚本。

```shell
python3.7 gen_dataset_info.py bin  ./val_FixRes ./prep_bin.info 384 384
```

“bin”：生成的数据集文件格式。

“./val_FixRes”：预处理后的数据文件的**相对路径**。

“./prep_bin.info”：生成的数据集文件保存的路径。

“384”：图片的宽和高。

运行成功后，在当前目录中生成“prep_bin.info”。

## 5 离线推理

### 5.1 benchmark工具概述

benchmark工具为华为自研的模型推理工具，支持多种模型的离线推理，能够迅速统计出模型在Ascend310上的性能，支持真实数据和纯推理两种模式，配合后处理脚本，可以实现诸多模型的端到端过程，获取工具及使用方法可以参考[CANN V100R020C10 推理benchmark工具用户指南 01](https://support.huawei.com/enterprise/zh/doc/EDOC1100164874?idPath=23710424%7C251366513%7C22892968%7C251168373)。

### 5.2 离线推理

1.设置环境变量

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.执行离线推理
增加benchmark.{arch}可执行权限

```shell
chmod u+x benchmark.x86_64
```

执行推理。执行时使npu-smi info查看设备状态，确保device空闲。

```shell
 ./benchmark.x86_64 -model_type=vision 
 -device_id=0 
 -batch_size=1 
 -om_path=./FixRes_bs1.om 
 -input_text_path=./prep_bin.info 
 -input_width=384 
 -input_height=384 
 -output_binary=False 
 -useDvpp=False
```

参数说明：

--model_type：模型类型。

--om_path：om文件路径。

--device_id：NPU设备编号。

--batch_size：参数规模。

--input_text_path：图片二进制信息。

--input_width：输入图片宽度。

--input_height：输入图片高度。

--useDvpp：是否使用Dvpp。

--output_binary：输出二进制形式。

推理后的输出默认在当前目录result下。

推理后的输出默认在当前目录“result/dumpOutput_device0”下。

## 6 精度对比

### 6.1 离线推理Acc精度统计

调用“FixRes_postprocess.py”脚本与数据集标签“imagenet_labels_fixres.json”比对，可以获得Top 1 Accuracy数据，结果保存在“result.json”中。

```shell
python3.7 FixRes_postprocess.py 
--label_file=./imagenet_labels_fixres.json 
--pred_dir=./result/dumpOutput_device0 > result.json
```

参数说明：

--label_file：生成推理结果所在路径。

--pred_dir：标签数据。

“result.json”：生成结果文件。

执行完后得到310P上的精度。

```
Top 1 Accuracy: 79.1%
```

### 6.2 精度对比

 **评测结果：**官网pth精度[rank1:79.0%](https://github.com/facebookresearch/FixRes)，  310离线推理精度rank1:79.1%。

## 7 性能对比

### 7.1 310性能数据

**性能测试：** 测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务。性能测试可使用`benchmark`工具。

```
./benchmark.x86_64 -round=20 -om_path=./FixRes_bs4.om -device_id=0 -batch_size=4
```

执行20次纯推理取均值，统计吞吐率与其倒数时延（benchmark的时延是单个数据的推理时间），npu性能是一个device执行的结果。

`benchmark`工具在整个数据集上推理方式测性能可能时间较长，纯推理方式测性能可能不准确，因此bs1要使用在整个数据集上推理的方式测性能。

```
./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path=./FixRes_bs1.om -input_text_path=./prep_bin.info -input_width=384 -input_height=384 -output_binary=False -useDvpp=False
```

**Interface throughputRate:** 183.263 * 4 = 733.052, 即是batch1 310单卡吞吐率。

### 7.2 310P性能数据

**Interface throughputRate:** 893.562, 即是batch1 310P单卡吞吐率。

### 7.3 T4性能数据

在装有T4卡的服务器上使用`onnxruntime-gpu`工具测试gpu性能，测试代码如下。测试过程请确保卡没有运行其他任务。

```python
import time
from turtle import width
import numpy as np
import onnxruntime as rt
print(rt.get_device())

batch_size=1
length=224
width=224
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

sess = rt.InferenceSession("./FixRes.onnx", providers=providers)
print("providers:",sess.get_providers())
input_name = sess.get_inputs()[0].name
outputs = ["output"]

# onnx模型输入节点
data = np.random.randn(batch_size,3, length, width).astype(np.float32)

# 推理200次，
for K in range(200):
    start_time = time.time()
    result = sess.run([], {input_name: data})
    end_time = time.time() - start_time
    time_list.append(end_time)

print("Batch_size: ",batch_size)
print("Time used: ", np.mean(time_list), 's')
print("T4 Throughput: ",batch_size/np.mean(time_list))
```

**T4 Throughput:** 200.616，即是batch1 T4单卡吞吐率。

### 7.4 性能对比

性能在310P上的性能达到310的1.2倍，达到T4性能的1.6倍，性能达标。
