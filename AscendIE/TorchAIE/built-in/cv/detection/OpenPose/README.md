# OpenPose模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)


******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

  本项目利用昇腾推理引擎`AscendIE`和框架推理插件`TorchAIE`，基于`pytorch框架`实现lightweight-human-pose-estimation模型在昇腾设备上的高性能推理。

   - 参考论文：[Daniil Osokin.Real-time 2D Multi-Person Pose Estimation on CPU:Lightweight OpenPose.(2018)](https://arxiv.org/abs/1811.12004)

   - 参考实现：

   ```
   url=https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
   branch=master 
   commit_id=1590929b601535def07ead5522f05e5096c1b6ac
   ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 大小                      | 数据类型 | 数据排布格式 |
  | -------- | ------------------------- | -------- | ------------ |
  | input    | batchsize x 3 x 368 x 640 | RGB_FP32 | NCHW         |


- 输出数据

  | 输出数据 | 大小                     | 数据类型 | 数据排布格式 |
  | -------- | ------------------------ | -------- | ------------ |
  | output1  | batchsize x 19 x 46 x 80 | FLOAT32  | ND           |
  | output2  | batchsize x 38 x 46 x 80 | FLOAT32  | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本      | 环境准备指导                                                 |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动 | 23.0.rc1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN | 7.0.RC1.alpha003 | - |
  | Python | 3.9.11 | - |
  | PyTorch | 2.0.1 | - |
  | Torch_AIE | 6.3.rc2 | - | - |

  
- 安装依赖

   ```
   pip install -r requirements.txt
   ```

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

    源码目录结构：

    ``` 
   ├── inference.py
   ├── preprocess.py
   ├── postprocess.py
   └── README.md
    ```

2. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git  
   cd lightweight-human-pose-estimation.pytorch
   git checkout master
   git reset --hard 1590929b601535def07ead5522f05e5096c1b6ac
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型使用coco val2017.zip验证集进行测试，数据集文件夹结构如下：

    ```
   coco
   ├──val2017
       ├── img.jpg
   ├──annotations
       ├── person_keypoints_val2017.json
       ...
    ```

2. 数据预处理。

   1. 建立数据存储文件夹。

      ```
      mkdir -p ./datasets/coco/processed_img
      mkdir -p ./output
      ```

   2. 数据预处理将原始数据集转换为模型输入的数据。

      执行“preprocess.py”脚本文件。

      ```
      python preprocess.py --raw-img-path /data/datasets/coco/val2017
      ```

      - 参数说明：
        - --raw-img-path：为数据集路径。

      预处理后的数据文件存放在./datasets/coco/processed_img目录下, 预处理的pad信息存放在./output/pad.txt中


## 模型推理<a name="section741711594517"></a>

1. 权重获取。

   获取权重文件[checkpoint_iter_370000.pth](https://pan.baidu.com/s/15BDVngC8XepdtlFZ5K8ZAw)，提取码k3w8，放在当前目录weights下。
   
   ```
   weights
   └── checkpoint_iter_370000.pth
   ```


2. 开始推理验证。

   1. 执行推理。
      ```
      python inference.py
      ```
      推理结果保存在./output/aie_inference_result目录，推理结束后会输出当前batch的推理性能。

   2. 精度验证。

      ```
      python postprocess.py --labels /data/datasets/coco/annotations/person_keypoints_val2017.json
      ```

      - 参数说明：
        - --labels ：标签数据。
   
   3. 性能测试
      ```
      python inference.py --input_shape 4 3 368 640
      ```
      - 参数说明：
        - --input_shape ：推理时输入shape, 可通过修改第1维测试不同batch的推理性能。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>


| 芯片型号 | Batch Size | 数据集 | 精度   | 性能     |
| -------- | ---------- | ------ | ------ | -------- |
| 310P3 | 1 | coco2017 | 0.404 | 671 FPS |
| 310P3 | 4 | coco2017 | 0.404 | 763 FPS |
| 310P3 | 8 | coco2017 | 0.404 | 677 FPS |
| 310P3 | 16 | coco2017 | 0.404 | 636 FPS |
| 310P3 | 32 | coco2017 | 0.404 | 621 FPS |
| 310P3 | 64 | coco2017 | 0.404 | 602 FPS |
