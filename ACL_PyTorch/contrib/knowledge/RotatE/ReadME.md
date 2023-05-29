# RotatE模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

`RotatE` 是一种知识图谱嵌入的新方法，该方法能够对知识图谱中的实体进行建模，并推断各种关系模式，包括：对称/反对称，反演和合成。具体来说，RotatE模型将每个关系定义为在复矢量空间中从源实体到目标实体的旋转。通过多个基准知识图上的实验，结果表明，RotatE模型不仅具有可伸缩性、而且还能够推断和建模各种关系模式，明显优于现有的用于链路预测的模型。

- 参考实现：

  ```
  url=https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
  branch=master
  commit_id=2e440e0f9c687314d5ff67ead68ce985dc446e3a
  code_path=KnowledgeGraphEmbedding
  model_name=KnowledgeGraphEmbedding
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小              | 数据排布格式 |
  | -------- | -------- | ----------------- | ------------ |
  | pos      | RGB_FP32 | batchsize x 3     | ND           |
  | neg      | RGB_FP32 | batchsize x 14541 | ND           |

- 输出数据

  > `-1` 表示动态shape含义

  | 输出数据 | 数据类型 | 大小    | 数据排布格式 |
  | -------- | -------- | ------- | ------------ |
  | score    | FLOAT32  | -1 x -1 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                            | 版本    | 环境准备指导                                                                                          |
  | --------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                      | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 5.1.RC1 | -                                                                                                     |
  | Python                                                          | 3.7.5   | -                                                                                                     |
  | PyTorch                                                         | 1.12.0  | -                                                                                                     |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码

    ```
    git clone https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding -b master
    cd KnowledgeGraphEmbedding
    git reset --hard 2e440e0f9c687314d5ff67ead68ce985dc446e3a
    cd ..
    ```

2. 安装依赖

     - 使用pip安装依赖
       ```
       pip install -r requirements.txt
       ```
     - 使用conda安装依赖（推荐）
       ```shell
       conda create --name <env> --file requirements.txt
       ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

    本模型所用在代码仓中 `KnowledgeGraphEmbedding/data/`。目录结构如下：
    ```
    KnowledgeGraphEmbedding/data/
    |-- FB15k
    |-- FB15k-237
    |-- YAGO3-10
    |-- countries_S1
    |-- countries_S2
    |-- countries_S3
    |-- wn18
    `-- wn18rr
    ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行 `RotatE_preprocess.py` 脚本，完成预处理。

   ```
   python rotate_preprocess.py --test_batch_size=1 --output_path='bin-bs1/'
   ```
   - 参数说明：
     - `--test_batch_size`：输入参数。
     - `--output_path`：输出文件路径。

    使用的数据集为FB15k-237，运行后生成“bin-bs1/”文件夹。支持后续不同bs的推理测试。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      从源码包中获取权重文件：[checkpoint](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/RotaE/PTH/checkpoint)。

   2. 导出onnx文件。

      使用 `RotatE_pth2onnx.py` 导出onnx文件。

      ```
      python rotate_pth2onnx.py --pth_path="./checkpoint" --onnx_path="./kge_onnx_head.onnx" --mode="head-batch"
      python rotate_pth2onnx.py --pth_path="./checkpoint" --onnx_path="./kge_onnx_tail.onnx" --mode="tail-batch"
      ```
      - 参数说明：
        - `--pth_path`：checkpoint文件的路径。
        - `--onnx_path`：生成的onnx的存放路径。
        - `--mode`：生成head-batch或tail-batch的onnx文件。
      获得 `kge_onnx_head.onnx` 和 `kge_onnx_tail.onnx` 文件。

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
          bs=1    # 以batchsize=1为例，其它batch自行修改
          atc --framework=5 \
              --model=kge_onnx_head.onnx \
              --output=kge_onnx_head \
              --input_format=ND \
              --input_shape="pos:$bs,3;neg:$bs,14541" \
              --log=error \
              --soc_version=Ascend${chip_name}

          atc --framework=5 \
              --model=kge_onnx_tail.onnx \
              --output=kge_onnx_tail \
              --input_format=ND \
              --input_shape="pos:$bs,3;neg:$bs,14541" \
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

              运行成功后生成 `kge_onnx_head.om` 模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```shell
      bs=1
      python -m ais_bench \
          --model ./kge_onnx_head.om \
          --input "./bin-bs1/head/pos,./bin-bs1/head/neg" \
          --output ./RotatEout/head \
          --output_dirname bs$bs \
          --outfmt NPY \
          --batchsize $bs

      python -m ais_bench \
          --model ./kge_onnx_tail.om \
          --input "./bin-bs1/tail/pos,./bin-bs1/tail/neg" \
          --output ./RotatEout/tail \
          --output_dirname bs$bs \
          --outfmt NPY \
          --batchsize $bs
      ```
      - 参数说明：
        - --model：模型地址
        - --input：预处理完的数据集文件夹
        - --output：推理结果保存地址
        - --outfmt：推理结果保存格式


   3. 精度验证。

      ```python
      python rotate_postprocess.py  \
                --head_result_path='./RotatEout/head/bs1' \
                --tail_result_path='RotatEout/tail/bs1' \
                --data_head='./bin-bs1/head' \
                --data_tail='./bin-bs1/tail' > result_bs1.json
      ```
      - 参数说明：
        - --result_path：推理结果对应的文件夹
        - --data_head：处理后的原始数据集--head
        - --data_tail：处理后的原始数据集--tail
      生成的精度结果在 `result_bs1.json` 文件中

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```shell
      bs=1
      python -m ais_bench \
          --model ./kge_onnx_head.om \
          --loop=20 \
          --batchsize $bs

      python -m ais_bench \
          --model ./kge_onnx_tail.om \
          --loop=20 \
          --batchsize $bs
      ```

      - 参数说明：
        - --model：模型地址
        - --loop：循环次数

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，在 `FB15k-237` 数据集上计算的精度结果为 `MRR:0.33568`。性能参考下列数据。

| 芯片型号 | Batch Size | 数据集    | 精度        | head性能 | tail性能 |
| -------- | ---------- | --------- | ----------- | ---- | ---- |
| 310P     | 1          | FB15k-237 | MRR:0.33568 |  179.32    |   177.95   |
| 310P     | 4          | FB15k-237 | MRR:0.33555 |   218.40   |   218.64   |
| 310P     | 8          | FB15k-237 | MRR:0.33555 |   221.14   |   221.22   |
| 310P     | 16         | FB15k-237 | MRR:0.33555 |   222.61   |   222.48   |
| 310P     | 32         | FB15k-237 | MRR:0.33555 |   222.79   |   222.76   |
| 310P     | 64         | FB15k-237 | MRR:0.33555 |   222.86   |   222.89   |
