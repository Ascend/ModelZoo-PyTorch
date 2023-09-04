# VGG16模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

   - [输入输出数据](#ZH-CN_TOPIC_0000001126281702)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

VGGNet是牛津大学计算机视觉组（Visual Geometry Group）和Google DeepMind公司的研究员一起研发的深度卷积神经网络，它探索了卷积神经网络的深度与其性能之间的关系，通过反复堆叠3*3的小型卷积核和2*2的最大池化层，成功地构筑了16~19层深的卷积神经网络。VGGNet相比之前state-of-the-art的网络结构，错误率大幅下降，VGGNet论文中全部使用了3*3的小型卷积核和2*2的最大池化核，通过不断加深网络结构来提升性能。
VGG16包含了16个隐藏层（13个卷积层和3个全连接层）

- 参考实现：

  ```shell
  url=https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
  branch=master
  commit_id=78ed10cc51067f1a6bac9352831ef37a3f842784
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | output1  | FLOAT32  | batchsize x 1000 | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                            | 版本    | 环境准备指导                                                                                          |
  | --------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                      | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 5.1.RC2 | -                                                                                                     |
  | Python                                                          | 3.7.13  | -                                                                                                     |
  | PyTorch                                                         | 1.9.0   | -                                                                                                     |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

- 该模型需要以下依赖

  **表 1**  依赖列表

  | 依赖名称        | 版本     |
  | --------------- | -------- |
  | torch           | 1.9.0    |
  | torchvision     | 0.10.0   |
  | onnx            | 1.9.0    |
  | onnx-simplifier | 0.3.6    |
  | onnxruntime     | 1.8.0    |
  | numpy           | 1.21.0   |
  | Cython          | 0.29.25  |
  | Opencv-python   | 4.5.4.60 |
  | pycocotools     | 2.0.3    |
  | Pytest-runner   | 5.3.1    |
  | protobuf        | 3.20.0   |
  | decorator       | \        |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码

1. 获取VGG16源代码。

   安装 torchvision 包即可

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（假设 `dataset_dir=/home/HwHiAiUser/dataset`）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。

   数据目录结构请参考：
   ```
   |-- dataset
     |-- ILSVRC2012_val_00000001.JPEG
     |-- ILSVRC2012_val_00000002.JPEG
     |-- ILSVRC2012_val_00000003.JPEG
     |-- ...
   ```

2. 数据预处理。

   将原始数据集转换为模型输入的二进制数据。执行“vgg16_preprocess.py”脚本。
   ```shell
   python vgg16_preprocess.py ${dataset_dir} ./prep_dataset
   ```
   - 参数说明
      - `${dataset_dir}`：原始数据验证集（.jpeg）所在路径
      - `./prep_dataset`：输出的二进制文件（.bin）所在路径

    每个图像对应生成一个二进制bin文件，一个附加信息文件。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      从源码包中获取权重文件：“vgg16-397923af.pth” 或者通过[下载链接](https://download.pytorch.org/models/vgg16-397923af.pth)

   2. 导出onnx文件。

      使用 `vgg16_pth2onnx.py` 导出onnx文件。

      ```shell
      python vgg16_pth2onnx.py --pth_path=./vgg16-397923af.pth --out=./vgg16.onnx
      ```

      获得 `vgg16.onnx` 文件。

   3. 使用ATC工具将ONNX模型转OM模型。

       1. 配置环境变量。

          ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
          ```

       2. 执行命令查看芯片名称（$\{chip\_name\}）。

          ```
          npu-smi info
          #该设备芯片名为Ascend310P3 （自行替换）
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
         # 这里以batchsize=1为例说明
         atc --framework=5 \
             --model=./vgg16.onnx \
             --output=vgg16_bs1 \
             --input_format=NCHW \
             --input_shape="actual_input_1:1,3,224,224" \
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

           运行成功后生成<u>**vgg16_bs1.om**</u>模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。
      ```shell
      python -m ais_bench \
            --model ./vgg16_bs1.om \
            --input ./prep_dataset \
            --output ./vgg16out/ \
            --outfmt TXT \
            --batchsize 1
      ```

      - 参数说明：

        -   --model：om文件路径。
        -   --input：预处理完的数据集文件夹
        -   --output：推理结果保存地址
        -   --outfmt：推理结果保存格式
        -   --batchsize：batchsize大小

      推理后的输出在 `--output` 所指定目录下。


   3. 精度验证。

      调用脚本与数据集标签 `val_label.txt` 比对，可以获得Accuracy数据。

      ```shell
      python vgg16_postprocess.py \
            --gtfile=./val_label.txt \
            --result_path=./vgg16out/2022_xx_xx-xx_xx_xx/sumary.json
      ```
      -   参数说明：
      -   --result_path：生成推理结果summary.json所在路径。
      -   --gtfile_path：标签val_label.txt所在路径

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

精度参考下列数据:
|       | top1_acc | top5_acc |
| ----- | -------- | -------- |
| 310   | 0.7128   | 0.9038   |
| 310P3 | 0.7128   | 0.9038   |
| 310B1 | 0.7128   | 0.9      |

性能参考下列数据:

|               | 310*4      | 310P3       | 310B1      |
| ------------- | ---------- | ----------- | ---------- |
| bs1           | 460.68     | 465.54      | 108.68     |
| bs4           | 834.48     | 1056.92     | 192.5      |
| bs8           | 947.02     | 1187.63     | 223.36     |
| bs16          | 1041.48    | 1424.51     | 243.59     |
| bs32          | 1076.3     | 1338.95     | 254.18     |
| bs64          | 936.44     | 1430.39     | 256.81     |
| **最优batch** | **1076.3** | **1424.51** | **256.81** |
