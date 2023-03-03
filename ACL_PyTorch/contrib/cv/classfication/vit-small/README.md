# VIT_small模型-推理指导

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

Vision Transformer是一个经典的图像分类网络。以前的cv领域虽然引入了transformer，但是都同时用到了cnn或者rnn。Vision Transformer直接使用纯transformer的结构并在图像识别上取得了不错的结果。本文档描述的是Vision Transformer中对配置为vit_small_patch16_224模型的Pytorch实现版本。

- 参考实现：

  ```shell
  url=https://github.com/rwightman/pytorch-image-models
  branch=main
  commit_id=a41de1f666f9187e70845bbcf5b092f40acaf097
  model_name=vision_transformer
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | output   | FLOAT32  | batchsize x 1000 | ND           |


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

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码

1. 获取VIT_small源代码。

    在已下载的源码包根目录下，执行如下命令。

    ```shell
    git clone https://github.com/rwightman/pytorch-image-models.git -b main
    cd pytorch-image-models/
    git reset --hard a41de1f666f9187e70845bbcf5b092f40acaf097
    patch -p1 < ../vit_small_patch16_224.patch
    cd ..
    ```

2. 安装依赖。

    ```
    pip install -r requirements.txt
    ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

    本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（假设 `dataset=/root/dataset/ILSVRC2012`）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。

    数据目录结构请参考：
    ```
    |-- ILSVRC2012
      |-- images
        |-- ILSVRC2012_val_00000001.JPEG
        |-- ILSVRC2012_val_00000002.JPEG
        |-- ILSVRC2012_val_00000003.JPEG
        |-- ...
      |-- val_label.txt
    ```

2. 数据预处理。

    将原始数据集转换为模型输入的二进制数据。执行 `vit_small_patch16_224_preprocess.py` 脚本。
    ```shell
    python vit_small_patch16_224_preprocess.py ${dataset}/images prep_dataset
    ```
    - 参数说明
      - 第一个参数是数据集路径
      - 第二个参数是生成的图片bin文件路径

    每个图像对应生成一个二进制bin文件。运行成功后，在当前目录下生成 `prep_dataset` 二进制文件夹和 `vit_prep_bin.info`。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      从源码包中获取权重文件：[S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Vitsmall/PTH/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz)，请将其放在与`vit_small_patch16_224_pth2onnx.py`文件同一目录内。

   2. 导出onnx文件。

      运行 `vit_small_patch16_224_pth2onnx.py` 脚本。

      ```shell
      python vit_small_patch16_224_pth2onnx.py \
        ./S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz \
        ./vit_small_patch16_224.onnx
      ```

      获得 `vit_small_patch16_224.onnx` 文件。

      - 参数说明：

        - 第一个参数是权重文件名称
        - 第二个参数是输出ONNX文件名称

   3. 使用ATC工具将ONNX模型转OM模型。

       1. 配置环境变量。

          ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
          ```

       2. 执行命令查看芯片名称（$\{chip\_name\}）。

          ```shell
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
          # 仅以batchsize=1进行说明
          atc --framework=5 \
              --model=vit_small_patch16_224.onnx \
              --output=vit_small_patch16_224_bs1 \
              --input_format=NCHW \
              --input_shape="image:1,3,224,224" \
              --log=error \
              --soc_version=Ascend${chip_name} \
              --enable_small_channel=1  \
              --optypelist_for_implmode="Gelu" \
              --op_select_implmode=high_performance
          ```

          - 参数说明：

            -   --model：为ONNX模型文件。
            -   --framework：5代表ONNX模型。
            -   --output：输出的OM模型。
            -   --input_format：输入数据的格式。
            -   --input_shape：输入数据的shape。
            -   --log：日志级别。
            -   --soc_version：处理器型号。
            -   --customize_dtypes：自定义算子的计算精度。
            -   --precision_mode：其余算子的精度模式。
            运行成功后生成 `vit_small_patch16_224_bs1.om` 模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。
      > `output` 路径根据用户需求自由设置，这里以 `output=./result` 为例说明
      ```shell
      python -m ais_bench \
          --model ./vit_small_patch16_224_bs1.om \
          --input ./prep_dataset/ \
          --batchsize 1 \
          --output ./result \
          --outfmt TXT
      ```

      -   参数说明：

          -   --model：om文件路径。
          -   --input:输入路径
          -   --output：输出路径。
          -   --outfmt：输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”。
          -   --loop：推理次数，可选参数，默认1，profiler为true时，推荐为1

         推理后的输出默认在 `--output` 文件夹下。


   3. 精度验证。

      调用脚本与数据集val2017标签比对。这里的 `dataset_path` 需要指定bin文件所在的路径，一般是 `dataset_path=${output}/2022_xxx` 这样的路径。同时需要删除 `$dataset_path` 目录下的 `sumary.json` 文件
      ```shell
      python vit_small_patch16_224_postprocess.py \
          ./result/2022_xxx/ \
          ${dataset}/val_label.txt \
          ./ \
          result.json
      ```

      - 参数说明：

        - --dataset_path：数据集路径。
        - --model_config：使用的开源代码文件路径。
        - --bin_data_path：推理结果所在目录。
        - --meta_info：数据预处理后获得的文件。
        - --net_out_num：输出节点数量。
        - --model_input_height：图片的高。
        - --model_input_width：图片的宽。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

1.精度对比

| 模型          | 仓库pth精度 | 310离线推理精度 | 310P离线推理精度 |
| ------------- | ----------- | --------------- | ---------------- |
| vit-small bs1 | top1:81.388 | top1:81.1       | top1:81.37       |
| vit-small bs8 | top1:81.388 | top1:81.1       | top1:81.37       |

2.性能对比

| Throughput | 310      | 310P     | T4       | 310P/310 | 310P/T4 |
| ---------- | -------- | -------- | -------- | -------- | ------- |
| bs1        | 203.054  | 435.2014 | 391.5258 | 2.14     | 1.11    |
| bs4        | 213.0816 | 771.9063 | 591.5261 | 3.62     | 1.30    |
| bs8        | 213.6788 | 1013.199 | 621.6682 | 4.74     | 1.63    |
| bs16       | 204.7552 | 913.2987 | 595.5638 | 4.46     | 1.53    |
| bs32       | 187.1448 | 778.4453 | 590.2469 | 4.15     | 1.31    |
| bs64       | 508.6636 | 730.0449 | 613.2265 | 1.19     | 1.19    |
|            |          |          |          |          |         |
| 最优batch  | 508.6636 | 1013.199 | 621.6682 | 1.99     | 1.63    |
