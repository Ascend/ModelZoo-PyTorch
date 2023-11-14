# ECAPA-TDNN模型-推理指导


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

ECAPA-TDNN基于人脸验证和计算机视觉相关领域的最新趋势，对传统的TDNN引入了多种改进。其中包括一维SE blocks，多层特征聚合（MFA）以及依赖于通道和上下文的统计池化。


- 参考实现：

  ```
  url=https://github.com/Joovvhan/ECAPA-TDNN.git
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FP32 | batchsize x 80 x 200 | ND         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 192 | ND           |
  | output2  | FLOAT32  | batchsize x 200 x 1536 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 23.0.0  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 7.0.0   | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone --recursive https://github.com/Joovvhan/ECAPA-TDNN.git
   mv ECAPA-TDNN ECAPA_TDNN
   export PYTHONPATH=$PYTHONPATH:./ECAPA_TDNN
   export PYTHONPATH=$PYTHONPATH:./ECAPA_TDNN/tacotron2
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   用户需自行获取VoxCeleb1数据集中测试集（无需训练集），上传数据集到服务器中,必须要与preprocess.py同目录。目录结构如下：

   ```
   VoxCeleb1
   ├── id10270
      ├── 1zcIwhmdeo4
         ├── 00001.wav 
         ├── ... 
   ├── id10271
   ├── ...
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   在当前工作目录下，执行以下命令行,其中VoxCeleb为数据集相对路径，input/为模型所需的输入数据相对路径，speaker/为后续后处理所需标签文件的相对路径

   ```
   python3 preprocess.py VoxCeleb1 input/ speaker/
   ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      ```
      wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Ecapa_tdnn/PTH/checkpoint.zip
      unzip checkpoint.zip
      ```
      该权重为自己训练出的权重，后续精度以该权重下精度为标准

      获取基准精度，作为精度对比参考， checkpoint为权重文件相对路径， VoxCeleb为数据集相对路径

      ```
      python3 get_originroc.py checkpoint VoxCeleb1
      ```

   2. 导出onnx文件。

      1. 使用pytorch2onnx.py导出onnx文件。

         利用权重文件和模型的网络结构转换出所需的onnx模型， checkpoint为权重文件相对路径， ecapa_tdnn.onnx 为生成的onnx模型相对路径。

         ```
         python3 pytorch2onnx.py checkpoint ecapa_tdnn.onnx 
         ```

         获得ecapa_tdnn.onnx文件。

      2. 优化ONNX文件。

         1. 安装onnx优化工具onnx_tool

            ```
            git clone https://gitee.com/zheng-wengang1/onnx_tools.git
            cd onnx_tools && git checkout cbb099e5f2cef3d76c7630bffe0ee8250b03d921
            cd ..
            ```
            
         2. 执行fix_conv1d.py

            ```
            python3 fix_conv1d.py ecapa_tdnn.onnx ecapa_tdnn_sim.onnx
            ```

         获得ecapa_tdnn_sim.onnx文件。

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

         ```
         atc --framework=5 --model=ecapa_tdnn_sim.onnx --output=./om/ecapa_tdnn_bs4 --input_format=ND --input_shape="mel:4,80,200" --log=debug  --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***ecapa_tdnn_bs4.om***</u>模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 配置环境变量。

      ```
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      ```

   3. 执行推理。

        ```
        python3 -m ais_bench --model "om/ecapa_tdnn_bs4.om" --input "input" --output "result" --output_dirname "output_bs4" --outfmt BIN
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入数据目录。
             -   --output：推理结果输出路径。
             -   --outfmt：推理结果输出格式。
            
        推理后的输出默认在当前目录result下。

   4. 精度验证。

      根据第四步中获取的结果result/output_bs4和第三步中产生的speaker标签文件，得到推理精度。

      ```
      python3 postprocess.py result/output_bs4 speaker
      ```

      - 参数说明：

        - result/output_bs4：为推理结果所在路径
        - speaker_bs4：为标签数据所在路径
        - 4：batch size
        - 4648：样本总数

   5. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
      ```

      - 参数说明：
        - --model：om文件路径。
        - --batchsize：batch大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

|roc_auc|   om   |   pth   |
|-------|--------|---------|
|bs1	  |0.9991  |   0.9991|
|bs4	  |0.9991  |   0.9989|

| Model      | batch_size | T4Throughput/Card | 310PThroughput/Card |
|------------|------------|-------------------|--------------------|
| ECAPA-TDNN | 1          | 485.43            | 981             |
| ECAPA-TDNN | 4          | 705.46            | 1654            |
| ECAPA-TDNN | 8          | 798.4             | 1461            |
| ECAPA-TDNN | 16         | 770.89            | 1302            |
| ECAPA-TDNN | 32         | 828.84            | 1211            |
| ECAPA-TDNN | 64         | 847.37            | 1238            |
| ECAPA-TDNN | best       | 847.37            | 1654            |