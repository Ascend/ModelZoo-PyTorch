# vovnet39模型-推理指导


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

基于VoVNet的目标检测模型性能超越基于DenseNet的模型，速度也更快，相比ResNet也是性能更好。

- 参考实现：

  ```shell
  url=https://github.com/AlexanderBurkhart/cnn_train.git
  commit_id=505637bcd08021e144c94e81401af6bc71fd46c6
  model_name=cnn_train
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小              | 数据排布格式 |
  | -------- | -------- | ----------------- | ------------ |
  | input    | FP32     | 1 x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output   | FLOAT32  | 1 x 1000 | ND           |

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
   git clone https://github.com/AlexanderBurkhart/cnn_train.git
	cd cnn_train
	git reset --hard 505637bcd08021e144c94e81401af6bc71fd46c6
	cd ..
   ```

2. 安装依赖。

   ```shell
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（假设 `dataset_dir=/home/HwHiAiUser/dataset`）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。

   数据目录结构请参考：
   ```
   |-- ILSVRC2012
     |-- ILSVRC2012_val_00000001.JPEG
     |-- ILSVRC2012_val_00000002.JPEG
     |-- ILSVRC2012_val_00000003.JPEG
     |-- ...
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   运行数据预处理脚本，将原始数据转换为符合模型输入要求的bin文件。
   ```shell
   python vovnet39_preprocess.py ${dataset_dir}/ILSVRC2012 ./prep_dataset
   ```
   -参数说明：
     + 第一个参数：验证集的路径
     + 第二个参数：输出文件的路径

    运行成功后，会在当前目录下生成二进制文件。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       该推理项目使用源码包中的权重文件[vovnet39.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/VoVnet39/PTH/vovnet39.pth)。

   2. 导出onnx文件。

      使用vovnet39_pth2onnx.py导出onnx文件。

      ```shell
      python vovnet39_pth2onnx.py  vovnet39.pth vovnet39.onnx
      ```

      获得vovnet39.onnx文件。


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
         # 本文以batch_size=1为例进行说明
         atc --framework=5 \
            --model=./vovnet39.onnx \
            --input_format=NCHW \
            --input_shape="image:1,3,224,224" \
            --output=vovnet39_bs1 \
            --log=debug \
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

        运行成功后生成vovnet39_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```shell
      python -m ais_bench \
              --model ./vovnet39_bs1.om \
              --input ./prep_dataset \
              --batchsize 1 \
              --outfmt TXT  \
              --output ./result
      ```

      - 参数说明：
         + --model: OM模型路径。
         + --input: 存放预处理bin文件的目录路径
         + --output: 存放推理结果的目录路径
         + --batchsize：每次输入模型的样本数
         + --outfmt: 推理结果数据的格式

        推理后的输出默认在当前目录result下。


   3. 精度验证。

      执行后处理脚本，计算 topN 精度：

      ```shell
      python vovnet39_postprocess.py \
                result/2022_10_26-15_44_59  \
                /opt/npu/ImageNet/val_label.txt \
                ./ \
                result.json
      ```

      - 参数说明：
         + 第一个参数: 推理输出目录
         + 第二个参数: 数据集配套标签
         + 第三个参数: 生成文件的保存目录
         + 第四个参数: 生成的文件名

      说明：精度验证之前，将推理结果文件中summary.json删除

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```shell
        python -m ais_bench --model vovnet39_bs1.om --loop 20 --batchsize 1
        ```

      -参数说明：
       + --model: om模型
       + --batchsize: 每次输入模型样本数
       + --loop: 循环次数

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

1. 精度对比

    | Model    | batchsize | Accuracy              | 开源仓精度            |
    | -------- | --------- | --------------------- | --------------------- |
    | vovnet39 | 1         | top1 76.77 top5 93.43 | top1 76.77 top5 93.43 |

2. 性能对比

    | batchsize | 310 性能 | T4 性能 | 310P 性能 | 310P/310 | 310P/T4 |
    | --------- | -------- | ------- | --------- | -------- | ------- |
    | 1         | 1128     | 657     | 832.6     | 0.7      | 1.2     |
    | 4         | 1318     | 1210.6  | 1767.6    | 1.3      | 1.4     |
    | 8         | 1375     | 1359.6  | 1756.6    | 1.3      | 1.2     |
    | 16        | 1483     | 1402.9  | 1728.6    | 1.16     | 1.2     |
    | 32        | 1483     | 1493.3  | 1703.5    | 1.12     | 1.14    |
    | 64        | 1219     | 1533.7  | 1551.2    | 1.2      | 1.01    |
