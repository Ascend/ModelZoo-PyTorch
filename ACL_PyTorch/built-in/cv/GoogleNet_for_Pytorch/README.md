# GoogleNet模型-推理指导

## 概述

  GoogLeNet是2014年Christian Szegedy提出的一种全新的深度学习结构，在这之前的AlexNet、VGG等结构都是通过增大网络的深度（层数）来获得更好的训练效果，但层数的增加会带来很多负作用，比如overfit、梯度消失、梯度爆炸等。inception的提出则从另一种角度来提升训练结果：能更高效的利用计算资源，在相同的计算量下能提取到更多的特征，从而提升训练结果 。 

-   参考论文：[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) 
-   参考实现：

```shell
url=https://github.com/pytorch/vision
branch=master
commit_id=7d955df73fe0e9b47f7d6c77c699324b256fc41f
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

  | 配套  | 版本 | 环境准备指导 |
  | ---- | ---- | ---------- |
  | 固件与驱动  | 22.0.1| [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN      | 6.0.RC1 |            |
  | PyTorch   | 1.13.1  |            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 |||


# 快速上手

## 获取源码

1. 获取源码。

   ```bsah
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/GoogleNet_for_Pytorch
   ```

   ```
   .
   |-- README.md
   |-- vision_metric_ImageNet.py      //验证推理结果脚本，比对模型输出的分类结果和标签，给出Accuracy
   |-- imagenet_torch_preprocess.py   //数据集预处理脚本
   |-- requirements.txt
   |-- googlenet_pth2onnx.py          //用于转换pth模型文件到onnx模型文件
   ```


2. 安装Python依赖。

   ```bash
   pip3 install -r requirments.txt
   ```

## 准备数据集

1. 获取原始数据集。

   本模型使用ImageNet 50000张图片的验证集，请前往ImageNet官网下载数据集

    ```
    ├── ImageNet
    |   ├── val
    |   |    ├── ILSVRC2012_val_00000001.JPEG
    │   |    ├── ILSVRC2012_val_00000002.JPEG
    │   |    ├── ......
    |   ├── val_label.txt
    ```

   

2. 数据预处理。

   执行预处理脚本，生成数据集预处理后的bin文件。

   ```bash
   python3 imagenet_torch_preprocess.py resnet ./ImageNet/val ./prep_dataset
   ```

   第一个参数为模型类型，第二个参数为原始数据验证集（.jpeg）所在路径，第三个参数为输出的二进制文件（.bin）所在路径。每个图像对应生成一个二进制文件。


## 模型推理

1. 模型转换。

   1. 获取模型权重文件。

      ```bash
      wget https://download.pytorch.org/models/googlenet-1378be20.pth
      ```

   2. 导出onnx文件。

      执行googlenet_pth2onnx.py脚本将.pth文件转换为.onnx文件，执行如下命令在当前目录生成```googlenet.onnx```模型文件。

      ```shell
      python3 googlenet_pth2onnx.py ./googlenet-1378be20.pth googlenet.onnx
      ```
      请访问[auto-optimizer推理工具](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)代码仓，根据readme文档进行工具安装。
      
      运行onnx_optimize.py脚本，优化模型。
      ```
      python3 onnx_optimize.py --model_path=googlenet.onnx --save_path=googlenet_opt.onnx
      ```
      生成优化后的onnx模型：googlenet_opt.onnx

   4. 使用ATC工具将ONNX模型转OM模型。

      1. 执行命令查看芯片名称（${chip_name}）。

         ${chip_name}可通过`npu-smi info`指令查看

          ![Image](https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/images/310P3.png)

      2. 使用atc将onnx模型转换为om模型文件。

         ```bash
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         
         # 请根据需要设置Batch size
         bs=8

         atc --framework=5 --model=./googlenet_opt.onnx --output=googlenet_bs${bs} --input_format=NCHW --input_shape="366:${bs},3,224,224" --log=debug --soc_version=${chip_name} --insert_op_conf=aipp.config --enable_small_channel=1
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

      运行成功后生成```googlenet_bs${bs}.om```文件。

   

2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。


   b.  执行推理。

   ```bash
   python3 -m ais_bench --model ./googlenet_bs${bs}.om --input ./prep_dataset/ --output ./ --output_dirname result --outfmt TXT
   ```

   -   参数说明：   
      - --model：模型地址
      - --input：预处理完的数据集文件夹
      - --output：推理结果保存地址
      - --outfmt：推理结果保存格式

   运行成功后会在```./result/```下生成推理输出的txt文件。

c.  精度验证。

调用vision_metric_ImageNet.py脚本与数据集标签val_label.txt比对，可以获得Accuracy Top5数据，结果保存在result.json中。

```shell
python3 vision_metric_ImageNet.py result/ ImageNet/val_label.txt ./ result.json
```
-   参数说明：
   - 第一个参数为生成推理结果所在路径
   - 第二个参数为标签数据
   - 第三个参数为生成结果文件路径
   - 第四个参数为生成结果文件名称



## 模型推理性能和精度


| 芯片型号     | Batch size | 精度 | 性能 |
| :---------: | :--------: |:------------------------:|:---------:|
| Ascend310P3 | 1          |Top1: 69.78%, Top5: 89.53%| 2333.64 |
| Ascend310P3 | 4          |                          | 4778.77 |
| Ascend310P3 | 8          |Top1: 69.78%, Top5: 89.53%| 6308.38 |
| Ascend310P3 | 16         |                          | 5906.61 |
| Ascend310P3 | 32         |                          | 5116.16 |
| Ascend310P3 | 64         |                          | 4886.97 |

