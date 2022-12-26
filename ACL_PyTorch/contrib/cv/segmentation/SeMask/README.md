# SeMask模型-推理指导

<!-- TOC -->

- [概述](#ZH-CN_TOPIC_0000001172161501)

  - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

<!-- /TOC -->

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

SeMask模型为第一个将语义上下文添加到预训练的Transformer主干以进行语义分割任务的模型。其通过两种技术将语义信息整合到通用的分层视觉转换器架构中，首先在Transformer Layer之后增加一个Semantic Layer；其次，使用了两个解码器：一个仅用于训练的轻量级语义解码器和一个特征解码器。

- 参考实现：

  ```text
  url=https://github.com/Picsart-AI-Research/SeMask-Segmentation
  commit_id=f12cf00c86afe669dfbea1dff0f8053ba49fed56
  model_name=SeMask-FPN
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型  | 大小                        | 数据排布格式 |
  | -------- | -------- | --------------------------- | ------------ |
  | image    | RGB_FP32 | batchsize x 3 x 1024 x 2048 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型                     | 大小    | 数据排布格式 |
  | -------- | ------- | ------- | ------------ |
  | output   | FP32 | batchsize x 1024 x 2048   | ND           |

# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.11.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```shell
    git clone https://github.com/Picsart-AI-Research/SeMask-Segmentation.git
    cd SeMask-Segmentation
    git reset --hard f12cf00c86afe669dfbea1dff0f8053ba49fed56
    cd ..
    mv SeMask-Segmentation/SeMask-FPN/ ./
   ```

    由于onnx不支持动态图，需要使用SeMask.patch修改模型代码。

    ```shell
    patch -p1 < SeMask.patch
    ```

    安装mmsegmentation依赖

    ```shell
    cd SeMask-FPN
    pip install -e .
    cd ..
    ```

2. 安装依赖。

   ```shell
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型支持cityscapes leftImg8bit的500张验证集。用户需要下载[leftImg8bit_trainvaltest.zip](http://www.cityscapes-dataset.com/downloads)和[gtFine_trainvaltest.zip](http://www.cityscapes-dataset.com/downloads)数据集，解压，将两个数据集放在./datasets/cityscapes/目录下。

    ```text
    .
    └──datasets
        └──cityscapes
            ├──gtFine
            |    ├──test
            |    ├──train
            |    └──val
            └──leftImg8bit
                ├──test
                ├──train
                └──val
    ```

2. 数据预处理。

    运行cityscapes脚本对数据进行预先处理

    ```shell
    python SeMask-FPN/tools/convert_datasets/cityscapes.py datasets/cityscapes/ --nproc 8
    ```

    运行SeMask_preprocess脚本将数据处理为bin文件

    ```shell
    python SeMask_preprocess.py \
    SeMask-FPN/configs/semask_swin/cityscapes/semfpn_semask_swin_small_patch4_window7_768x768_80k_cityscapes.py \
    --data_root ./datasets/cityscapes/ \
    --save_path ./preprocess_result/
    ```

    参数说明：

    - data_root：cityscapes数据文件存储的路径，可自行设置，默认值为./datasets/cityscapes/；
    - save_path：导出bin文件存储的路径，可自行设置，默认值为./preprocess_result。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

    使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

    1. 获取权重文件。

       获取权重文件方法，可从[SeMask-S FPN](https://drive.google.com/file/d/1WyT207dZmdwETBUR6aeiqOVfQdUIV_fN/view?usp=sharing)下载获取权重文件semask_small_fpn_cityscapes.pth

    2. 导出onnx文件。

        1. 使用SeMask_pth2onnx.py导出onnx文件。
            运行SeMask_pth2onnx.py脚本。

            ```shell
            python SeMask_pth2onnx.py \
            SeMask-FPN/configs/semask_swin/cityscapes/semfpn_semask_swin_small_patch4_window7_768x768_80k_cityscapes.py \
            --checkpoint semask_small_fpn_cityscapes.pth \
            --batch_size ${bs} \
            --output_file SeMask_bs${bs}.onnx
            ```

            参数说明：
            - checkpoint：pth权重文件的路径，可自行设置，默认值为semask_small_fpn_cityscapes.pth；
            - batch_size：导出的onnx模型的batch_size，可自行设置，默认值为1。
            - output_file：需要转出的onnx模型的名称，可自行设置，默认值为SeMask.onnx（由于本模型不支持动态batch，推荐在模型名后加后缀，如‘_bs1’，用以区分不同batch_size的onnx模型。)

        2. 使用onnxsim，简化onnx模型结构

            ```shell
            python -m onnxsim SeMask_bs${bs}.onnx SeMask_bs${bs}_sim.onnx
            ```

    3. 使用ATC工具将ONNX模型转OM模型。

        1. 配置环境变量。

            ```shell
            source /usr/local/Ascend/ascend-toolkit/set_env.sh
            ```

            > **说明：**
            >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

        2. 执行命令查看芯片名称。

            ```shell
            npu-smi info
            
            回显如下：
            +-------------------+-----------------+------------------------------------------------------+
            | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
            | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
            +===================+=================+======================================================+
            | 0       310P3     | OK              | 16.8         53                0    / 0              |
            | 0       0         | 0000:86:00.0    | 0            944  / 21534                            |
            +===================+=================+======================================================+
            ```

        3. 执行ATC命令。

            ```shell
            atc --framework=5 \
            --model=SeMask_bs${bs}_sim.onnx \
            --output=SeMask_bs${bs}  \
            --input_format=NCHW \
            --input_shape="image:${bs},3,1024,2048" \
            --soc_version=Ascend${chip_name} \
            ```

            - 参数说明：
              - --framework：5代表ONNX模型。
              - --model：为ONNX模型文件。
              - --output：输出的OM模型。
              - --input\_format：输入数据的格式。
              - --input\_shape：输入数据的shape。
              - --soc\_version：处理器型号。

            运行成功后生成om模型文件，推荐在模型名后加后缀，如‘_bs1’，用以区分不同batch_size的om模型。

2. 开始推理验证。

    1. 安装ais_bench推理工具。

        请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。  

    2. 执行推理。

        ```shell
        mkdir output
        python -m ais_bench \
        --model SeMask_bs${bs}.om \
        --batchsize ${bs} \
        --input ./preprocess_result/leftImg8bit/ \
        --output ./output \
        --outfmt BIN
        ```

        - 参数说明：
          - model：om文件路径。
          - batchsize：om文件对应的模型batch size。
          - input：模型输入的路径。
          - output：推理结果输出路径。
          - outfmt：输出数据的格式。

    3. 精度验证。

        后处理统计mIoU

        调用SeMask_postprocess.py脚本将推理结果与label进行比对，获取pixAcc和mIoU数据

        ```shell
        python SeMask_postprocess.py \
        SeMask-FPN/configs/semask_swin/cityscapes/semfpn_semask_swin_small_patch4_window7_768x768_80k_cityscapes.py \
        --input_dir ./preprocess_result/leftImg8bit/ \
        --result_dir ./output/${time_line}/ \
        --data_root ./datasets/cityscapes/
        ```

        - 参数说明：
          - input_dir：模型输入数据路径。
          - result_dir：模型输出文件路径。
          - data_root：数据集文件路径。

    4. 性能验证。

        可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```shell
        python -m ais_bench \
        --model=SeMask_bs{bs}.om \
        --loop=100 \
        --batchsize=${bs}
        ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号  | Batch Size       | 数据集      | 精度       | 性能      |
| --------- | ---------------- | ---------- | ---------- | --------- |
| 310P3     | 1                | cityscapes | 76.54      |  4.72     |
| 310P3     | 4                | cityscapes | 76.54      |  4.37   |
| 310P3     | 8                | cityscapes | 76.54      |  4.13     |
| 310P3     | 16                | cityscapes |  内存不足     |       |