# Focal-Transformer模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  ******

  

# 概述

Focal-Transformer是一个可用于图像分类、目标检测和语义分割的图像Transformer网络，有Tiny、Small以及Base三种规模可以选择。模型由标准图像Transformer改进而来，使用粗粒度和细粒度两种模式分别汇聚远距离和近距离的token信息，并且使用多尺度的金字塔结构，避免了分辨率平方倍复杂度的同时尽可能聚合信息。


- 参考实现：

  ```
  url=https://github.com/microsoft/Focal-Transformer
  branch=master
  commit_id=57bb3031582a2afb2d2a6916612bc4311316f9fc
  model_name=Focal-fast-S
  ```

## 输入输出数据

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | 1 x 1000 | ND           |


# 推理环境准备\[所有版本\]

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                    | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.13  | -                                                            |
| PyTorch                                                      | 1.8.1   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手

## 获取源码

1. 获取源码。

   ```
   git clone https://github.com/microsoft/Focal-Transformer.git
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集

1. 获取原始数据集。

   本模型采用ImageNet验证集。

   数据集：[ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)

   推理只需验证集即Validation images (all tasks) 以及对应的标签文件val_label.txt。

   存放路径：./ImageNet

   目录结构：

   ```
   ├── ImageNet
       ├── val
       └── val_label.txt
   ```

   

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行Focal_Transformer_preprocess.py脚本，完成预处理。

   ```shell
   python3 Focal_Transformer_preprocess.py \
   --input_path ImageNet/val \
   --output_path infer/databin/
   ```

   - 参数说明：
      - input_path：验证集路径。
      - output_path：输出bin文件路径。


## 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件

      [FocalTransformer模型权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/FocalvTransformer/PTH/focalv2-small-useconv-is224-ws7.pth)

   2. 导出onnx文件。

      1. 使用Focal_Transformer_pth2onnx.py导出onnx文件。

         ```shell
         python3 Focal_Transformer_pth2onnx.py \
                 --code_path ./Focal-Transformer \
                 --input_path ./focalv2-small-useconv-is224-ws7.pth \
                 --output_path ./focalv2.onnx
         ```
         
         - 参数说明：
            - code_path：模型源码文件所在路径。
            - input_path：预训练pth模型路径。
            - output_path：输出onnx模型路径
         
         获得focalv2.onnx文件。
   
   3. 使用ATC工具将ONNX模型转OM模型。
   
      1. 配置环境变量。
   
         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
   
         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
   
      2. 执行命令查看芯片名称（${chip_name}） 。
   
         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 
         回显如下：
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 15.8         42                0    / 0              |
         | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
         +===================+=================+======================================================+
         | 1       310P3     | OK              | 15.4         43                0    / 0              |
         | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
         +===================+=================+======================================================+
         ```
   
      3. 执行ATC命令。
         ```shell
         atc --framework=5 \
         --model=./focalv2.onnx \
         --output=./focalv2_bs1 \
         --input_format=NCHW \
         --input_shape="image:1,3,224,224" \
         --log=error \
         --soc_version=Ascend${chip_name}
         ```
         
         - 参数说明：
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
         
         运行成功后生成`focalv2_bs1.om`模型文件。



2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。
   
   2. 执行推理。

      ```shell
      python3 -m ais_bench \
               --model ./focalv2_bs1.om \
               --input infer/databin/ \
               --output infer/ \
               --output_dirname output \
               --outfmt TXT
      ```
   
      - 参数说明：
         - model：om模型路径。
         - input：bin数据集路径。
         - output：推理结果路径。
         - output_dirname：推理结果子目录。
         - outfmt：推理结果格式。

   
   3. 精度验证。
   
      调用脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。
      
      ```shell
      python3 Focal_Transformer_postprocess.py \
            --input_path infer/output \
            --label_path ImageNet/val_label.txt \
            --output_path infer/result.json
      ```
      
      - 参数说明：
      
         - input_path：推理结果路径。
         - label_path：标签路径。
         - output_path：精度验证json结果。
   
      json文件包含每张图片的测试结果以及当前的平均精度
   
   
   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model=focalv2_bs1.om --loop=20 --batchsize=${batch_size}
      ```

      - 参数说明：
        - --model：om模型文件路径。
        - --batchsize：模型输入对应的batch size。



# 模型推理性能&精度

调用ACL接口推理计算，精度仅支持Batch Size为1的情况，性能参考下列数据。

| 芯片型号      | Batch Size | 数据集     | 精度    | 性能    |
| ------------ | ---------- | ---------- | ------- | ------- |
|    310P3     | 1          | ImageNet1k | 83.586% |  7.96   |

模型仅支持batch size为1