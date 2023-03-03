# RetinaNet模型-推理指导

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

论文提出了一个简单、灵活、通用的损失函数Focal loss，用于解决单阶段目标检测网络检测精度不如双阶段网络的问题。这个损失函数是针对了难易样本训练和正负样本失衡所提出的，使得单阶段网络在运行快速的情况下，获得与双阶段检测网络相当的检测精度。此外作者还提出了一个Retinanet用于检验网络的有效性，其中使用Resnet和FPN用于提取多尺度的特征。

- 参考实现：

  ```shell
  url=https://github.com/facebookresearch/detectron2
  commit_id=60fd4885d7cfd52d4267d1da9ebb6b2b9a3fc937
  code_path=detectron2
  model_name=detectron2
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小              | 数据排布格式 |
  | -------- | -------- | ----------------- | ------------ |
  | input    | FP32     | 1 x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小    | 数据排布格式 |
  | -------- | -------- | ------- | ------------ |
  | boxes    | FLOAT32  | 5 x 100 | ND           |
  | labels   | INT64    | 1 x 100 | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套       | 版本    | 环境准备指导                                                                                          |
  | ---------- | ------- | ----------------------------------------------------------------------------------------------------- |
  | 固件与驱动 | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN       | 5.1.RC2 | -                                                                                                     |
  | Python     | 3.7.5   | -                                                                                                     |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```shell
   git clone https://github.com/facebookresearch/detectron2 -b main
   cd detectron2
   git reset --hard 60fd4885d7cfd52d4267d1da9ebb6b2b9a3fc937
   patch -p1 < ../Retinanet.diff
   pip install -e .
   cd -
   ```

2. 安装依赖。

   ```shell
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型需要coco2017数据集，数据集下载[地址](https://cocodataset.org/)

   受开源代码仓的限制，建议把coco数据集存放在 `detectron2/datasets` 目录下，并设置环境变量 `export DETECTRON2_DATASETS=detectron2/datasets`。

   其中val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json，文件目录结构如下：
   ```
    detectron2
    ├── datasets
    │   ├── coco
    │   │   ├── annotations
    │   │   ├── val2017
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

	 运行数据预处理脚本，将原始数据转换为符合模型输入要求的bin文件。
   ```shell
   python Retinanet_preprocess.py \
          --image_src_path=detectron2/datasets/coco/val2017 \
          --bin_file_path=val2017_bin \
          --model_input_height=1344 \
          --model_input_width=1344
   ```
    - 参数说明：
       + --image_src_path：原始数据验证集（.jpg）所在路径。
       + --bin_file_path：输出的二进制文件（.bin）所在路径。
       + --model_input_height：模型输入图像高度像素数量。
       + --model_input_width：模型输入图像宽度像素数量。

    运行成功后，会在当前目录下生成二进制文件。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       [RetinaNet-detectron2.pkl](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Retinanet/PTH/RetinaNet-detectron2.pkl)

   2. 导出onnx文件。

      ```shell
      python detectron2/tools/deploy/export_model.py \
            --config-file detectron2/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml \
            --output ./ \
            --export-method tracing \
            --format onnx MODEL.WEIGHTS RetinaNet-detectron2.pkl MODEL.DEVICE cpu
      ```

      获得model.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```shell
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
         batch_size=1             # 本文仅以batch_size=1为例进行说明
         atc --model=model.onnx \
            --framework=5 \
            --output=retinanet_bs${batch_size} \
            --input_format=NCHW \
            --input_shape="input0:$batch_size,3,1344,1344" \
            --out_nodes="Cast_1229:0;Reshape_1223:0;Gather_1231:0" \
            --log=error \
            --soc_version=Ascend${chip_name}
         ```

         - 参数说明：
            + --model: ONNX模型文件所在路径。
            + --framework: 5 代表ONNX模型。
            + --input_format: 输入数据的排布格式。
            + --input_shape: 输入数据的shape。
            + --output: 生成OM模型的保存路径。
            + --log: 日志级别。
            + --soc_version: 处理器型号。

        运行成功后生成 `retinanet_bs1.om` 模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```shell
      python -m ais_bench \
            --model ./retinanet_bs1.om \
            --input ./val2017_bin \
            --output ./ \
            --output_dirname result \
            --batchsize 1 \
            --outfmt BIN
      ```

      - 参数说明：
         + --model: OM模型路径。
         + --input: 存放预处理bin文件的目录路径
         + --output: 存放推理结果的目录路径
         + --output_dirname: 存放推理结果文件夹
         + --batchsize：每次输入模型的样本数
         + --outfmt: 推理结果数据的格式

        推理后的输出默认在当前目录result下。


   3. 精度验证。
      > 说明：精度验证之前，将推理结果文件中summary.json删除
      运行get_info.py脚本，生成图片数据info文件。

      ```shell
       python get_info.py jpg ./datasets/coco/val2017 val2017.info
      ```

      - 参数说明：
         + 第一个参数为生成的数据集文件格式
         + 第二个参数为原始数据文件相对路径
         + 第三个参数为生成的info文件名

      运行成功后，在当前目录生成val2017.info，执行后处理脚本，计算 map 精度：
      ```shell
      python Retinanet_postprocess.py \
            --bin_data_path=./result/ \
            --val2017_path=./datasets/coco \
            --test_annotation=val2017.info \
            --det_results_path=./ret_npuinfer/ \
            --net_out_num=3 \
            --net_input_height=1344 \
            --net_input_width=1344
      ```

      - 参数说明：
         + --bin_data_path: 推理结果所在路径
         + --val2017_path: 数据集所在路径
         + --test_annotation: 原始图片信息文件
         + --det_results_path: 后处理输出结果
         + --net_out_num: 网络输出个数
         + --net_input_height: 网络高
         + --net_input_width: 网络宽

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```shell
      python -m ais_bench --model retinanet_bs1.om --loop 20 --batchsize 1
      ```

      -参数说明：
       + --model: om模型
       + --batchsize: 每次输入模型样本数
       + --loop: 循环次数

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

1. 精度对比

    | Model     | batchsize | Accuracy    | 开源仓精度  |
    | --------- | --------- | ----------- | ----------- |
    | Retinanet | 1         | map = 38.3% | map = 38.6% |

2. 性能对比

    | batchsize | 310 性能 | T4 性能 | 310P 性能 | 310P/310 | 310P/T4 |
    | --------- | -------- | ------- | --------- | -------- | ------- |
    | 1         | 8.9      | 8.6     | 17        | 1.91     | 1.97    |
