# Ernie3 模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******





# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Ernie通过训练数据中的词法结构，语法结构，语义信息从而进行统一的建模，这极大地增强了通用语义表示能力。而Ernie3.0能读取更多的文字，并且同时支持NLG和NLU任务。



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input_ids    | int64 | batchsize x max_seq_len | ND         |
  | token_type_ids    | int64 | batchsize x max_seq_len | ND         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output  | FLOAT32  | batchsize x 2 | ND           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>


1. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       ```
       python3 export_onnx.py --model_name ernie-3.0-base-zh --model_type AutoModelForSequenceClassification --save_path ernie/
       ```
       - 参数说明
         - model_name: 模型名称
         - model_type: 模型类型
         - save_path: 模型权重保存文件夹

   2. 导出onnx文件。

      1. 运行命令。

         ```
         paddle2onnx --model_dir ernie/ --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file model.onnx --opset_version 11
         ```
         - 参数说明
            - model_dir: 模型权重文件夹
            - model_filename: 模型名称
            - params_filename: 模型权重
            - save_file：保存模型名称
            - opset_version：模型版本

         获得`model.onnx`文件。


   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         会显如下：
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

         ```
          atc --model=model.onnx \
               --framework=5 \
               --output=model \
               --input_format=ND \
               --input_shape="token_type_ids:8,128;input_ids:8,128" \
               --log=error \
               --soc_version=Ascend${chip_name} \
               --op_precision_mode=op_precision.ini
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --op_precision_mode: 使能op的性能模式

           运行成功后生成`model.om`模型文件。

2. 开始推理验证。
   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais-bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

        ```
        python3 infer.py --task_name csl --model_path model.om --use_pyacl 1 --device npu --device_id 0 --batch_size 8 --model_name_or_path ernie-3.0-base-zh
        ```

        -   参数说明：

             -   task_name：任务类型
             -   model_path：om模型
             -   use_pyacl：是否使用pyacl推理
             -   device：芯片型号
             -   device_id：芯片ID
             -   batch_size：模型bs
             -   model_name_or_path：模型名称

        推理后的精度结果默认保存在当前文件`results.txt`

   4. 性能验证。

      可使用ais-bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型
        - --batchsize：模型batchsize
        - --loop: 循环次数



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度（Acc） | 性能 |
| :------: | :--------: | :----: | :--: | :--: |
|    310P3      |     1       |    Clue    |  49%    |  517    |
|    310P3      |     4       |    Clue    |  49%    |  1044    |
|    310P3      |     8       |    Clue    |  49%    |  1313    |
|    310P3      |     16       |    Clue    |  49%    |  1238    |
|    310P3      |     32       |    Clue    |  49%    |  1097    |
|    310P3      |     64       |    Clue    |  49%    |  1096    |