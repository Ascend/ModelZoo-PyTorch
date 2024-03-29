# InceptionV3 模型推理指导

- [概述](#概述)
    - [输入输出数据](#输入输出数据)
- [推理环境](#推理环境)
- [快速上手](#快速上手)
    - [获取源码](#获取源码)
    - [准备数据集](#准备数据集)
    - [模型转换](#模型转换)
    - [推理验证](#推理验证)
- [性能&精度](#性能精度)

----
# 概述

InceptionV3 模型是谷歌 Inception 系列里面的第三代模型，在 InceptionV2 模型的基础上，InceptionV3 通过分解卷积和新的正则化方法，极大地减少了计算开销。

+ 论文  
    [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)  
    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna

+ 参考实现：  
    https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py

## 输入输出数据
模型输入

| input-name | data-type | data-format |input-shape |
| ---------- | --------- | ----------- | ---------- |
| input1     | FLOAT32   | ND          | bs x 3 x 299 x 299 |

模型输出

| output-name |  data-type | data-format |output-shape |
| ----------- | ---------- | ----------- | ----------- |
| output1      |  FLOAT32   | ND          | bs x 1000        |


----
# 推理环境

- 该模型推理所需配套的软件如下：

| 配套                    | 版本              | 
|-----------------------|-----------------| 
| CANN                  | 6.3.RC2.alph002 | -                                                       |
| Python                | 3.9.0           |                                                           
| PyTorch               | 2.0.1           |
| torchVison            | 0.15.2          |-
| Ascend-cann-torch-aie | -               
| Ascend-cann-aie       | -               
| 芯片类型                  | Ascend310P3     | -                                                         |

    
    说明：请根据推理卡型号与 CANN 版本选择相匹配的固件与驱动版本。


----
# 快速上手

## 安装

- 安装推理过程所需的依赖
    ```bash
    pip install -r requirements.txt
    ```

## 准备数据集

1. 获取原始数据集  
    本模型推理项目使用 ILSVRC2012 数据集验证模型精度，请在 [ImageNet官网](https://gitee.com/link?target=http%3A%2F%2Fimage-net.org%2F) 自行下载，并按照以下的目录结构存放图片与标签文件。   
    ```
    ├── imageNet/
        ├── val/
            ├──ILSVRC2012_val_00000001.JPEG
            ├──ILSVRC2012_val_00000002.JPEG
            ├──...
        ├── val_label.txt
    ```

## 推理验证
1. 下载PyTorch官方提供的[ **预训练模型** ](https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth) 到当前目录，可参考命令：
    ```
    wget https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth
    ```
    下载完成后，将模型权重文件放到当前目录下。

2. 数据预处理
    ```
    python3 inceptionv3_preprocess.py --src_path /path/to/val --save_path prep_dataset
    ```
   
3. 运行模型评估脚本，测试ImageNet验证集推理精度和性能
    ```
    python3 inceptionv3_eval.py --device_id 0 --prep_dataset prep_dataset --checkpoint inception_v3_google-1a9a5a14.pth 
        --label_file /path/to/val_label.txt --batch_size 1
    ```


----
# 模型推理性能及精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用torch-aie推理计算，精度参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度                             | 性能       |
| --------- |------------| ---------- |--------------------------------|----------|
| 310P3 | 1          | ImageNet | top-1: 77.302% <br>top-5: 93.464% | 1058 FPS |
| 310P3 | 4          | ImageNet |  top-1: 77.302% <br>top-5: 93.464% | 2269 FPS |
| 310P3 | 8          | ImageNet | top-1: 77.302% <br>top-5: 93.464% | 2309 FPS |
| 310P3 | 16         | ImageNet | top-1: 77.302% <br>top-5: 93.464% | 2066 FPS |
| 310P3 | 32         | ImageNet |top-1: 77.302% <br>top-5: 93.464%  | 2032 FPS |
| 310P3 | 64         | ImageNet |  top-1: 77.302% <br>top-5: 93.464% | 2007 FPS |

