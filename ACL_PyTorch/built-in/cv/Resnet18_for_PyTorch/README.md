# Resnet18模型-推理指导

## 概述

 ResNet是ImageNet竞赛中分类问题效果较好的网络，它引入了残差学习的概念，通过增加直连通道来保护信息的完整性，解决信息丢失、梯度消失、梯度爆炸等问题，让很深的网络也得以训练。ResNet有不同的网络层数，常用的有18-layer、34-layer、50-layer、101-layer、152-layer。 

### 1.1 论文地址
[ResNet18论文](https://arxiv.org/pdf/1512.03385.pdf)  

### 1.2 代码地址
[ResNet18代码](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)  

### 输入输出数据
  
  - 输入数据
    
    | 输入数据 | 大小                      | 数据类型 | 数据排布格式 |
    | -------- | ------------------------- | -------- | ------------ |
    |   image  | batchsize x 3 x 256 x 256 | RGB_FP32 |    NCHW      |
  
  - 输出数据
    
    | 输出数据 | 大小             | 数据类型 | 数据排布格式 |
    | -------- | ---------------- | -------- | ------------ |
    |   class  | batchsize x 1000 |  FLOAT32 |       ND     |


## 推理环境准备

- 该模型需要以下插件与驱动
  
  | 配套                                        | 版本      | 环境准备指导                                                                                        |
  | ----------------------------------------- | ------- | --------------------------------------------------------------------------------------------- |
  | 固件与驱动                                     | 22.0.4  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                      | 6.0.RC1 |                                                                                               |
  | PyTorch                                   | 1.5.1   |                                                                                               |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 |         |                                                              |

- 该模型需要以下依赖。

  | 依赖名称      | 版本   |
  | ------------- | ------ |
  | onnx          | 1.9.0  |
  | torch         | 1.5.1  |
  | torchVision   | 0.6.1  |
  | numpy         | 1.19.2 |
  | Pillow        | 8.2.0  |
  | opencv-python | 4.5.2  |



# 快速上手

#### 获取源码

1. 源码上传到服务器任意目录（如：/home/HwHiAiUser）。

   ```
   .
   |-- README.md
   |-- imagenet_acc_eval.py             //验证推理结果脚本，比对benchmark输出的分类结果和标签，给出Accuracy
   |-- imagenet_torch_preprocess.py     //数据集预处理脚本
   |-- requirements.txt
   |-- resnet18_pth2onnx.py             //用于转换pth模型文件到onnx模型文件
   ```
   
   

2. 请用户根据依赖列表和提供的requirements.txt以及自身环境准备依赖。

   ```
   pip3 install -r requirements.txt
   ```
   

## 准备数据集

本模型使用ImageNet 50000张图片的验证集，请前往[ImageNet官网](https://image-net.org/download.php)下载数据集:

```
├── ImageNet
|   ├── val
|   |    ├── ILSVRC2012_val_00000001.JPEG
│   |    ├── ILSVRC2012_val_00000002.JPEG
│   |    ├── ......
|   ├── val_label.txt
```

执行预处理脚本:

```bash
python3 imagenet_torch_preprocess.py --data_path ./ImageNet/val/ --save_dir ./prep_dataset
```
参数说明：
- --data_path：原始数据验证集val目录路径
- --save_dir：输出的二进制文件（.bin）所在路径
   

## 模型推理

1. 模型转换。

   本模型基于开源框架PyTorch训练的Resnet18进行模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      从PyTorch开源框架获取经过训练的Resnet18权重文件```resnet18-f37072fd.pth```
      ```bash
      wget https://download.pytorch.org/models/resnet18-f37072fd.pth
      ```

   2. 导出onnx文件。

      将.pth文件转换为.onnx文件，执行如下命令在当前目录生成```resnet18.onnx```模型文件。
      
      ```bash
      python3 resnet18_pth2onnx.py --checkpoint ./resnet18-f37072fd.pth --save_dir ./resnet18.onnx
      ```
      参数说明：
      - --checkpoint：模型权重pth文件
      - --save_dir：输出的onnx模型的路径

      

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 执行命令查看芯片名称（${chip_name}）。

         ${chip_name}可通过`npu-smi info`指令查看

          ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

      2. 使用atc将onnx模型转换为om模型文件。
   
         ```bash
         # 设置环境变量
         source /usr/local/Ascend/ascend-toolkit/set_env.sh

         # bs为批次大小，请根据需要设置
         bs=8
         
         atc --framework=5 --model=./resnet18.onnx --output=resnet18_bs${bs} --input_format=NCHW --input_shape="image:${bs},3,224,224" --log=debug --soc_version=${chip_name} --insert_op_conf=aipp.config --enable_small_channel=1
         ```
   
         参数说明：

         - --model：为ONNX模型文件。
         - --framework：5代表ONNX模型。
         - --output：输出的OM模型。
         - --input_format：输入数据的格式。
         - --input_shape：输入数据的shape。
         - --log：日志级别。
         - --soc_version：处理器型号，支持Ascend310系列。
         - --enable_small_channel：是否使能small channel的优化，使能后在channel<=4的卷积层会有性能收益。
         - --insert_op_conf=aipp.config: AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。

   

2. 开始推理验证。

   a.  安装ais_bench推理工具。

   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   b.  执行推理。

   ```bash
   python3 -m ais_bench --model ./resnet18_bs${bs}.om --input ./prep_dataset/ --output ./result/ --output_dir resnet18_bs${bs} --outfmt TXT
   ```

   参数说明：   
   - --model：模型地址
   - --input：预处理完的数据集文件夹
   - --output：推理结果保存地址
   - --outfmt：推理结果保存格式

   运行成功后会在result/resnet18_bs${bs}下生成推理输出的txt文件。

   c.  精度验证。

   调用imagenet_acc_eval.py脚本与数据集标签val_label.txt比对，可以获得Accuracy Top5数据，结果保存在result.json中。

   ```bash
   python3 imagenet_acc_eval.py result/resnet18_bs${bs} ./ImageNet/val_label.txt ./ result.json
   ```

   第一个参数为生成推理结果所在路径，第二个参数为标签数据，第三个参数为生成结果文件路径，第四个参数为生成结果文件名称。

3. 执行纯推理验证性能。
   ```bash
   python3 -m ais_bench --model resnet18_bs${bs}.om --device 1 --loop 100
   ```
   - 参数说明：
      - --model：om文件路径
      - --device：NPU设备编号
      - --loop: 纯推理次数

## 模型推理性能和精度

官方的参考精度为：Top1: 69.758%， Top5: 89.078%

| 芯片型号    | Batch size | 精度                     | 性能(fps) |
| :---------: | :--------: |:------------------------:|:---------:|
| Ascend310P3 | 1          |Top1: 69.75%, Top5: 89.10%|  2864.96  |
| Ascend310P3 | 4          |                          |  6821.99  |
| Ascend310P3 | 8          |                          |  9450.12  |
| Ascend310P3 | 16         |                          |  9705.72  |
| Ascend310P3 | 32         |                          |  9678.07  |
| Ascend310P3 | 64         |                          |  9816.71  |
| Ascend310P3 | 128        |Top1: 69.75%, Top5: 89.10%|  9827.83  |
| Ascend310P3 | 256        |                          |  9803.72  |