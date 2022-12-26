# Dino_Resnet50模型-推理指导

- [Dino\_Resnet50模型-推理指导](#dino_resnet50模型-推理指导)
- [概述](#概述)
- [推理环境](#推理环境)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型转换](#模型转换)
- [精度\&性能](#精度性能)

---

# 概述

Dino是Facebook于今年发表的最新的无监督学习成果，在图像处理分类等方面取得了很好的成果，而与经典的Resnet50的分类模型的残差单元相结合训练，经验证也依然保障了较高精度，与纯Resnet50模型相比精度基本没有下滑，同时也保持了性能。

- 论文

    [Caron, Mathilde, et al. "Emerging properties in self-supervised vision transformers." arXiv preprint arXiv:2104.14294 (2021).](https://arxiv.org/abs/2104.14294)


- 参考实现

    ```
    url = https://github.com/facebookresearch/dino
    branch = main
    commit_id = cb711401860da580817918b9167ed73e3eef3dcf
    ```

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32  | batchsize x 3 x 224 x 224 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | output   | FLOAT32  | batchsize x 1000 | ND           |


# 推理环境

- 该模型需要以下插件与驱动

    | 配套       | 版本    | 环境准备指导                                                                                           |
    | ---------- | ------- | ------------------------------------------------------------------------------------------------------ |
    | 固件与驱动 | 22.0.2  | [Pytorch 框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN       | 5.1.RC2 | -                                                                                                      |
    | Python     | 3.7.5   | -                                                                                                      |

---

# 快速上手

## 获取源码


1. 下载本仓，复制该推理项目所在目录，进入复制好的目录
    ```
    git clone https://github.com/facebookresearch/dino
    cd dino
    git reset --hard cb711401860da580817918b9167ed73e3eef3dcf
    cd ..
    ```

2. 执行以下命令安装所需的依赖
    ```shell
    pip install -r requirements.txt
    ```

## 准备数据集

1. 获取原始数据集

   本模型使用 [ImageNet官网](http://www.image-net.org) 的5万张验证集进行测试。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（假设 `dataset_dir=/home/HwHiAiUser/dataset`）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。

   数据目录结构请参考：
   ```
   |-- dataset
     |-- ILSVRC2012_val_00000001.JPEG
     |-- ILSVRC2012_val_00000002.JPEG
     |-- ILSVRC2012_val_00000003.JPEG
     |-- ...
   ```

2. 数据预处理

    运行数据预处理脚本，将原始数据转换为符合模型输入要求的bin文件。
    ```shell
    python dino_resnet50_preprocess.py ${dataset_dir} prep_dataset
    ```
    参数说明：
    + ${datasets_path}: 原始数据验证集（.jpeg）所在路径。
    + prep_dataset: 输出的二进制文件（.bin）所在路径。

    运行成功后，会在当前目录下生成二进制文件。

## 模型转换


1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      从源码包中获取权重文件：[dino_resnet50_pretrain.pth](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth) 和[dino_resnet50_linearweights.pth](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_linearweights.pth)

   2. 导出onnx文件。

      使用 `dino_resnet50_pth2onnx.py` 导出onnx文件。

      ```shell
      python dino_resnet50_pth2onnx.py \
                --backbone_pth=./dino_resnet50_pretrain.pth  \
                --linear_pth=./dino_resnet50_linearweights.pth  \
                --out=./dino_resnet50.onnx
      ```

      获得 `dino_resnet50.onnx` 文件。

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
             --model=./dino_resnet50.onnx \
             --output=dino_resnet50_bs1 \
             --input_format=NCHW \
             --input_shape="input:1,3,224,224" \
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

           运行成功后生成<u>**dino_resnet50_bs1.om**</u>模型文件。

2. 推理验证

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   2. 执行推理。
      ```shell
      python -m ais_bench \
            --model ./dino_resnet50_bs1.om \
            --input ./prep_dataset \
            --output ./dinoresnet50_out/ \
            --output_dirname bs1 \
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
      >**说明：**
      >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见 --help命令。

   3. 精度验证。

      调用脚本与数据集标签 `val_label.txt` 比对，可以获得Accuracy数据。

      ```shell
      python dino_resnet50_postprocess.py \
            --gtfile=${dataset_dir}/val_label.txt \
            --result_path=./dinoresnet50_out/sumary.json
      ```
      -   参数说明：
      -   --result_path：生成推理结果summary.json所在路径。
      -   --gtfile_path：标签val_label.txt所在路径

# 精度&性能

1. 精度对比

    | Model         | batchsize | Accuracy                                      | 开源仓精度                                    |
    | ------------- | --------- | --------------------------------------------- | --------------------------------------------- |
    | dino_resnet50 | 1         | top1 accuracy = 75.28% top5 accuracy = 92.56% | top1 accuracy = 75.28% top5 accuracy = 92.56% |

2. 性能对比
    | batchsize | 310 性能     | T4 性能     | 310P 性能  | 310P/310 | 310P/T4 |
    | --------- | ------------ | ----------- | ---------- | -------- | ------- |
    | 1         | 1617.052 fps | 878.742 fps | 1378.7 fps | 0.85     | 1.6     |
    | 4         | 2161.044 fps | 1532.6 fps  | 5539.4 fps | 2.5      | 3.6     |
    | 8         | 2410.1 fps   | 1733.5fps   | 10986 fps  | 4.5      | 6.3     |
    | 16        | 2441.2 fps   | 1858.1fps   | 22119 fps  | 9        | 11      |
    | 32        | 5279.8fps    | 2033.1fps   | 43852fps   | 8        | 21.5    |
    | 64        | 2244fps      | 2090fps     | 87537fps   | 39       | 41      |