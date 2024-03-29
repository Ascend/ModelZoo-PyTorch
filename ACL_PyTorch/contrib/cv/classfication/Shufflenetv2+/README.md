# Shufflenetv2+模型-推理指导

## 概述

   在对于小型网络的性能评价标准中常用的是计算复杂度FLOPs，在这其中还有一个非常关键的评价标准是Speed运行速度，这通常收到计算机内存存储和运行平台等因素的影响。本文提出了一个新的网络ShuffleNet v2，对卷积网络在轻量级优化的参数量和运行速度等方面进行研究分析。 

-   参考论文：[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)
-   参考实现：

```shell
url=https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2%2B
branch=master
commit_id=d69403d4b5fb3043c7c0da3c2a15df8c5e520d89
```



### 输入输出数据

- #### 输入输出数据

  - 输入数据

    | 输入数据 | 大小                      | 数据类型 | 数据排布格式 |
    | -------- | ------------------------- | -------- | ------------ |
    | input    | batchsize x 3 x 224 x 224 | RGB_FP32 | NCHW         |

  - 输出数据

    | 输出数据 | 大小             | 数据类型 | 数据排布格式 |
    | -------- | ---------------- | -------- | ------------ |
    | output1  | batchsize x 1000 | FLOAT32  | ND           |


### 推理环境准备

- 该模型需要以下插件与驱动

  | 配套                                                         | 版本                                                         | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 固件与驱动                                                   | [1.0.15](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | [5.1.RC1](https://www.hiascend.com/software/cann/commercial?version=5.1.RC1) |                                                              |
  | PyTorch                                                      | [1.5.1](https://github.com/pytorch/pytorch/tree/v1.5.1)      |                                                              |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 |                                                              |                                                              |

- 该模型需要以下依赖。

  | 依赖名称      | 版本     |
  | ------------- | -------- |
  | onnx          | >=1.9.0  |
  | torch         | >=1.5.1  |
  | torchVision   | >=0.6.1  |
  | numpy         | >=1.19.2 |
  | Pillow        | >=8.2.0  |
  | opencv-python | >=4.5.2  |



## 快速上手

### 获取源码

1. 源码上传到服务器任意目录（如：/home/HwHiAiUser）。

   ```
   .
   |-- README.md
   |-- imagenet_acc_eval.py             //验证推理结果脚本，比对模型输出的分类结果和标签，给出Accuracy
   |-- imagenet_torch_preprocess.py     //数据集预处理脚本
   |-- requirements.txt
   |-- shufflenetv2_pth2onnx.py         //用于转换pth模型文件到onnx模型文件
   ```

   

2. 请用户根据依赖列表和提供的requirments.txt以及自身环境准备依赖。

   ```
   pip3 install -r requirments.txt
   ```

   

### 准备数据集

1. 获取原始数据集。

   本模型使用ImageNet官网的5万张验证集进行测试，请用户自行获取该数据集，上传数据集到服务器任意目录（如：*/home/HwHiAiUser/dataset*）。图片与标签分别存放在*/home/HwHiAiUser/dataset*/imagenet/val与*/home/HwHiAiUser/dataset*/imagenet/val_label.txt位置。

   

2. 数据预处理。

   执行预处理脚本，生成数据集预处理后的bin文件。

   ```shell
   python3 imagenet_torch_preprocess.py /home/HwHiAiUser/dataset/imagenet/val ./prep_dataset
   ```

   第一个参数为模型类型，第二个参数为原始数据验证集（.jpeg）所在路径，第三个参数为输出的二进制文件（.bin）所在路径。每个图像对应生成一个二进制文件。

   

### 模型推理

1. 模型转换。

   本模型基于开源框架PyTorch训练的ShuffleNetV2+进行模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      点击[Link](https://pan.baidu.com/share/init?surl=EUQVoFPb74yZm0JWHKjFOw)在PyTorch开源框架获中取经过训练的ShuffleNetV2+权重文件ShuffleNetV2+.Small.pth.tar，源码中已提供下载权重文件 。

      **备注**：百度网盘提取码：mc24

   2. 下载shufflenetv2+模型代码。

      ```shell
      git clone https://github.com/megvii-model/ShuffleNet-Series.git
      ```
      
   3. 导出onnx文件。

      shufflenetv2_pth2onnx.py脚本将.pth文件转换为.onnx文件，执行如下命令在当前目录生成shufflenetv2_bs1.onnx模型文件。

      ```shell
      python3 shufflenetv2_pth2onnx.py shufflenetv2.pth shufflenetv2_bs1.onnx 1
      ```

      使用ATC工具将.onnx文件转换为.om文件，导出.onnx模型文件时需设置算子版本为11。

        参数说明：

        - 第一参数为输入的pth模型。

        - 第二参数为输出的onnx模型名。

        - 第三参数为batchsize。

          **注意：**

          >动态batch的onnx转换om失败并且测的性能数据也不对，每个batch的om需要对应batch的onnx来转换，每个batch的性能数据需要对应batch的模型来测试

   4. 使用ATC工具将ONNX模型转OM模型。

      1. 执行命令查看芯片名称（${chip_name}）。

         ${chip_name}可通过`npu-smi info`指令查看

          ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

      2. 使用atc将onnx模型转换为om模型文件。

         ```shell
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
           
         atc --framework=5 --model=./shufflenetv2_bs1.onnx --input_format=NCHW --input_shape="image:1,3,224,224" --output=shufflenetv2_bs1 --log=debug --soc_version=Ascend${chip_name}
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

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
python3 -m ais_bench --model ./resnet101_bs1.om --input ./prep_dataset/ --output ./result/ --outfmt TXT
```

-   参数说明：   
    --model：模型地址
    --input：预处理完的数据集文件夹
    --output：推理结果保存地址
    --outfmt：推理结果保存格式

运行成功后会在result/xxxx_xx_xx-xx-xx-xx（时间戳）下生成推理输出的txt文件。


c.  精度验证。

调用imagenet_acc_eval.py脚本与数据集标签val_label.txt比对，可以获得Accuracy Top5数据，结果保存在result.json中。

```shell
python3 imagenet_acc_eval.py result/xxxx_xx_xx-xx-xx-xx（时间戳） /home/HwHiAiUser/datasets/imagenet/val_label.txt ./ result.json
```

第一个参数为生成推理结果所在路径，第二个参数为标签数据，第三个参数为生成结果文件路径，第四个参数为生成结果文件名称。



