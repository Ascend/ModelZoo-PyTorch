# Shufflenetv2模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

本项目利用昇腾推理引擎`AscendIE`和框架推理插件`TorchAIE`，基于`pytorch框架`实现Shufflenetv2模型在昇腾设备上的高性能推理。



- 参考实现：

  ```
  https://github.com/pytorch/vision/blob/v0.14.0/torchvision/models/shufflenetv2.py
  ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input0    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型        | 大小 | 数据排布格式 |
  | -------- | ------------ | -------- | ------------ |
  | output1  | RGB_FP32 | batchsize x 3 x 224 x 224  | NCHW           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套  | 版本  | 环境准备指导  |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动 | 23.0.rc1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN | 7.0.RC1.alpha003 | - |
  | Python | 3.9.11 | - |
  | PyTorch | 2.0.1 | - |
  | Torch_AIE | 6.3.rc2 | - |

- 安装依赖

   ```
   pip install -r requirements.txt
   ```


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型使用ImageNet验证集进行推理测试 ，用户自行获取数据集后，将文件解压并上传数据集到任意路径下。数据集目录结构如下所示：

   ```
   imageNet/
   |-- val
   |   |-- ILSVRC2012_val_00000001.JPEG
   |   |-- ILSVRC2012_val_00000002.JPEG
   |   |-- ILSVRC2012_val_00000003.JPEG
   |   ...
   |-- val_label.txt
   ...
   ```



## 模型推理<a name="section741711594517"></a>

  1. 导出ts模型

```
python3 export.py
```


  2. 精度验证。

```
python3 run.py --data_path ./datasets/ImageNet_50000/val
```

    - 参数说明：

      - data_path：数据保存路径
  
  3. 性能测试。

```
python3 perf.py --mode=ts --batch_size=1
```
    - 参数说明：
      - batch_size ：推理时输入shape, 可通过修改第1维测试不同batch的推理性能。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| Ascend310P3 | 1 | ImageNet | 69.33% | 1437 FPS |
| Ascend310P3 | 4 | ImageNet | 69.33% | 3245 FPS |
| Ascend310P3 | 8 | ImageNet | 69.33% | 4374 FPS |
| Ascend310P3 | 16 | ImageNet | 69.33% | 4986 FPS |
| Ascend310P3 | 32 | ImageNet | 69.33% | 7201 FPS |
| Ascend310P3 | 64 | ImageNet | 69.33% | 6356 FPS |
