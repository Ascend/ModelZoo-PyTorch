# EfficientDetD0 ONNX模型端到端推理指导

- [1 模型概述](#1-模型概述)
    -   [1.1 论文地址](#11-论文地址)
    -   [1.2 代码地址](#12-代码地址)
- [2 环境说明](#2-环境说明)
    -   [2.1 深度学习框架](#21-深度学习框架)
    -   [2.2 python第三方库](#22-python第三方库)
- [3 模型转换](#3-模型转换)
    -   [3.1 pth转onnx模型](#31-pth转onnx模型)
    -   [3.2 onnx转om模型](#32-onnx转om模型)
- [4 数据集预处理](#4-数据集预处理)
    -   [4.1 数据集获取](#41-数据集获取)
    -   [4.2 数据集预处理](#42-数据集预处理)
- [5 离线推理](#5-离线推理)
    -   [5.1 构建ais_infer工具](#51-构建ais_infer工具)
    -   [5.2 离线推理](#52-离线推理)
- [6 精度对比](#6-精度对比)
    -   [6.1 离线推理精度](#61-离线推理精度)
    -   [6.2 精度对比](#62-精度对比)
- [7 性能对比](#7-性能对比)
    - [7.1 获取npu性能](#71-获取npu性能)
    - [7.2 获取T4性能](#72-获取T4性能)
    - [7.3 性能数据](#73-性能数据)
    - [7.4 性能对比](#74-性能对比)





## 1 模型概述

EfficientDet该论文首先提出了一种加权双向特征金字塔网络（BiFPN），它允许简单、快速的多尺度特征融合；其次，提出了一种复合特征金字塔网络缩放方法，统一缩放所有backbone的分辨率、深度和宽度、特征网络和box/class预测网络。

当融合不同分辨率的特征时，一种常见的方法是首先将它们调整到相同的分辨率，然后将它们进行总结。金字塔注意网络global self-attention上采样恢复像素定位。所有以前的方法都一视同仁地对待所有输入特征。 然而，论文中认为由于不同的输入特征在不同的分辨率，他们通常贡献的输出特征不平等。为了解决这个问题，论文建议为每个输入增加一个权重，并让网络学习每个输入特性的重要性。

### 1.1 论文地址

[Tan M, Pang R, Le Q V . EfficientDet: Scalable and Efficient Object Detection[C] 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020.](https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html)

### 1.2 代码地址

```shell
ur=https://github.com/rwightman/efficientdet-pytorch
branch=master
commit_id=c5b694aa34900fdee6653210d856ca8320bf7d4e
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
timm == 0.6.1
pyyaml == 5.3.1
numpy == 1.20.0
pycocotools == 2.0.4
omegaconf == 2.2.2
onnx-simplifier == 0.4.5
skl2onnx == 1.10.4
backports.lzma == 0.0.14
```

**说明：** 

>   X86架构：pytorch，torchvision和onnx可以通过官方下载whl包安装，其它可以通过pip3.7 install 包名 安装
>
>   Arm架构：pytorch，torchvision和onnx可以通过源码编译安装，其它可以通过pip3.7 install 包名 安装

## 3 模型转换

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

### 3.1 pth转onnx模型

1. 准备pth权重文件  
   使用训练好的pkl权重文件：d0.pth

   下载地址： [链接: https://pan.baidu.com/s/1rDt4I9yobrApJqQFS13n3A  密码: 1f47](https://pan.baidu.com/s/1rDt4I9yobrApJqQFS13n3A)

   请在EfficientDetD0目录下创建model文件夹，并将d0.pth文件移入model文件夹中

2. 安装efficientdet-pytorch与onnx_tools

   ```
   git clone https://github.com/rwightman/efficientdet-pytorch.git
   cd efficientdet-pytorch
   git checkout c5b694aa34900fdee6653210d856ca8320bf7d4e
   patch -p1 < ../effdet.patch
   cd ..
   git clone https://gitee.com/zheng-wengang1/onnx_tools.git
   cd onnx_tools
   git checkout cbb099e5f2cef3d76c7630bffe0ee8250b03d921
   patch -p1 < ../onnx_tools.patch
   cd ../
   ```

3. 导出onnx文件

   1. 转出原始onnx文件

      ```
      python3.7 pth2onnx.py --batch_size=1 --checkpoint=./model/d0.pth --out=./model/d0.onnx 
      ```

      参数说明：

      - batch_size:根据需求选择batch大小
      - checkpoint：pytorch权重文件  
      - out：输出onnx模型  

   2. 精简优化网络

      ```
      python3.7 -m onnxsim --input-shape="1,3,512,512" --dynamic-input-shape ./model/d0.onnx ./model/d0_sim.onnx --skip-shape-inference
      ```

      这里的input-shape 第一个维度需要与上文batch_size保持一致

   3. 将部分pad算子constant_value值-inf修改为0

      ```
      python3.7 modify_onnx.py --model=./model/d0_sim.onnx --out=./model/d0_modify.onnx
      ```

   最终，会在./model/目录下生成d0_modify.onnx作为最终得到的onnx文件

   

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
atc --framework=5 \
--model=./model/d0_modify.onnx \
--output=./model/d0 \
--input_format=NCHW \
--input_shape="x.1:1,3,512,512" \
--log=debug \
--soc_version=Ascend${chip_name} \
--precision_mode=allow_mix_precision \
--modify_mixlist=ops_info.json 
```

参数说明：
--model：为ONNX模型文件

--framework：5代表ONNX模型

--output：输出的OM模型

--input_format：输入数据的格式

--input_shape：输入数据的shape

--log：日志级别

--soc_version：处理器型号

--precision_mode：由于float16在部分batch_size下存在溢出情况，故需开启混合精度

--modify_mixlist：混合精度配置文件

## 4 数据集预处理

### 4.1 数据集获取

本模型支持coco2017 val 5000张图片的验证集。请用户自行获取数据集，上传数据集到代码仓目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到coco val2017.zip验证集及~/annotations中的instances_val2017.json数据标签。

数据目录结构请参考：

```
coco_data
├──val2017
├── annotations
    ├── instances_val2017.json
```

请将coco_data文件夹放在EfficientDetD0目录下。

### 4.2 数据预处理

数据预处理将原始数据集转换为模型输入的数据。

执行“preprocess.py”脚本，完成预处理。

```shell
python3.7 preprocess.py --root=coco_data --bin-save=bin_save
```

--root：coco数据集文件

--bin-save：输出的二进制文件（.bin）所在路径

每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成bin_save文件夹。

## 5 离线推理

### 5.1 构建[ais_infer工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)

1. 本推理工具编译需要安装好CANN环境。用户可以设置CANN_PATH环境变量指定安装的CANN版本路径，比如export CANN_PATH=/xxx/nnae/latest/. 如果不设置，本推理工具默认会从 CANN_PATH /usr/local/Ascend/nnae/latest/ /usr/local/Ascend/ascend-toolkit/latest 分别尝试去获取
2. 请克隆ais_infer代码仓库并进入ais-bench/tool/ais_infer目录，执行如下命令进行编译，即可生成推理后端whl包

```
git clone https://gitee.com/ascend/tools.git
cd  tools/ais-bench_workload/tool/ais_infer/backend
pip3.7 wheel ./
```

3. 在运行设备上执行如下命令，进行安装

```
pip3.7 install ./aclruntime-0.0.1-cp37-cp37m-linux_aarch64.whl
```

如果安装提示已经安装了相同版本的whl，请执行命令请添加参数"--force-reinstall"

```
pip3.7 install ./aclruntime-0.0.1-cp37-cp37m-linux_aarch64.whl --force-reinstall
```

4. 当whl包安装完成后，请在EfficientDetD0目录下进行如下操作:

```
git clone https://gitee.com/ascend/tools.git
cd tools
mv ais-bench_workload/tool/ais_infer/ais_infer.py ais-bench_workload/tool/ais_infer/frontend/ ../
cd ../
```



### 5.2 离线推理

1.设置环境变量

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2.执行离线推理
执行推理前请使用npu-smi info查看设备状态，确保device空闲。

```shell
rm -rf ./result && mkdir result
python3.7 ais_infer.py \
--model model/d0.om \
--input ./bin_save \
--output ./result \
--outfmt BIN \
--batchsize=1 \
--infer_queue_count=1
```

参数说明：

--model：模型类型。

--input：经过预处理后的bin文件路径

--output：输出文件路径

--outfmt：输出文件格式

--batchsize：批次大小

--infer_queue_count: 推理队列的数据最大数

推理后的结果会保存在./result目录下

## 6 精度对比

### 6.1 离线推理精度

调用“postprocess.py”脚本即可获得最终mAP精度：

```shell
python3.7 postprocess.py --root=./coco_data --omfile=./result
```

参数说明：

--root：coco数据集路径

--omfile：模型推理结果

执行完后得到310P上的精度。

```
mAP: 33.4
```

对于batch_size=1，4，8，16，32，64，精度均如上。

### 6.2 精度对比

 **评测结果：**官网pth精度[mAP: 33.6](https://github.com/rwightman/efficientdet-pytorch)，  310离线推理精度mAP: 33.4，精度下降在1%范围之内，故精度达标。

## 7 性能对比

### 7.1 获取npu性能

**性能测试：** 测试npu性能要确保device空闲，使用npu-smi info命令可查看device是否在运行其它推理任务。性能测试可使用`ais_infer`工具。

```
python3.7 ais_infer.py --model model/d0.om --output ./ --outfmt BIN --loop 20 --batchsize=1
```

可以直接从**throughtput**中读取模型对应吞吐率。

`ais_infer`工具在整个数据集上推理方式测性能可能时间较长，纯推理方式测性能可能不准确，因此模型要使用在整个数据集上推理的方式测性能。

```
python3.7 ais_infer.py --model model/d0.om --input ./bin_save --output ./result --outfmt BIN --batchsize=1 --infer_queue_count=1
```

### 7.2 获取T4性能

在装有T4卡的服务器上使用`onnxruntime-gpu`工具测试gpu性能，测试代码如下。测试过程请确保卡没有运行其他任务。

```python
import time
from turtle import width
import numpy as np
import onnxruntime as rt
print(rt.get_device())

batch_size=1
length=512
width=512
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

sess = rt.InferenceSession("./model/d0_modify.onnx", providers=providers)
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

### 7.3 性能数据

| Throughput | 310    | 310P    | T4     | 310P/310    | 310P/T4     |
| ---------- | ------ | ------- | ------ | ----------- | ----------- |
| bs1        | 47.256 | 67.074  | 9.945  | 1.419375317 | 6.744494721 |
| bs4        | 53.564 | 135.378 | 12.284 | 2.527406467 | 11.0206773  |
| bs8        | 54.116 | 143.944 | 12.332 | 2.659915737 | 11.67239702 |
| bs16       | 54.761 | 145.109 | 12.377 | 2.649860302 | 11.724085   |
| bs32       | 54.001 | 142.913 | 12.388 | 2.646488028 | 11.5364062  |
| bs64       | 54.136 | 143.258 | 12.476 | 2.646261268 | 11.48268676 |
| 最优batch  | 54.761 | 145.109 | 12.476 | 2.649860302 | 11.63105162 |

### 7.4 性能对比

性能在310P上的性能达到310的1.2倍，达到T4性能的1.6倍，性能达标。