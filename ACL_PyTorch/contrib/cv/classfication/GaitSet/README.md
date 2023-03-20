# GaitSet模型-推理指导


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

GaitSet是一个灵活、有效和快速的跨视角步态识别网络，迁移自https://github.com/AbnerHqC/GaitSet



- 参考实现：

  ```
  url=https://github.com/AbnerHqC/GaitSet
  commit_id=14ee4e67e39373cbb9c631d08afceaf3a23b72ce
  model_name=GaitSet
  ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 100 x 64 x 44 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型        | 大小 | 数据排布格式 |
  | -------- | ------------ | -------- | ------------ |
  | output1  | FLOAT32 | batchsize x 62 x 256  | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.5.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |




# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/AbnerHqC/GaitSet.git
   cd GaitSet
   git reset --hard 14ee4e67e39373cbb9c631d08afceaf3a23b72ce
   git apply ../change.patch
   ```

2. 安装依赖。

   ```
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   本模型支持CASIA-B图片的验证集。下载地址http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp  ，只下载DatasetB数据集。

   下载后的数据集内的压缩文件需要全部解压，解压后数据集内部的目录应为（`GaitDatasetB-silh`数据集）：数据集路径/对象序号/行走状态/角度，如
   ```
   GaitDatasetB-silh
   ├── 001      
   └── 002
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行GaitSet_preprocess_step1.py脚本

   ```
   python GaitSet_preprocess_step1.py --input_path=./GaitDatasetB-silh --output_path=./predata
   ```
   -   参数说明：

         -   input_path：数据集地址
         -   output：初步预处理保存地址

   执行GaitSet_preprocess_step2.py脚本，完成预处理
   ```
   mkdir CASIA-B-bin
   python GaitSet_preprocess_step2.py --data_path=./predata --bin_file_path=./CASIA-B-bin/
   ```   
   -   参数说明：

         -   data_path：初步预处理结果
         -   bin_file_path：预处理数据地址



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
      源码仓中存在权重文件，地址是GaitSet/work/checkpoint/GaitSet/GaitSet_CASIA-B_73_False_256_0.2_128_full_30-80000-encoder.ptm

   2. 导出onnx文件。

      1. 使用GaitSet_pth2onnx.py导出onnx文件。

         运行GaitSet_pth2onnx.py脚本。

         ```
         python GaitSet_pth2onnx.py --input_path=./GaitSet/work/checkpoint/GaitSet/GaitSet_CASIA-B_73_False_256_0.2_128_full_30-80000-encoder.ptm
         ```

         获得XXX.onnx文件。


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
         atc --framework=5 \
             --model=gaitset_submit.onnx \
             --output=gaitset_submit_bs${bs} \
             --input_shape="image_seq:${bs},100,64,44" \
             --log=debug \
             --soc_version=Ascend{chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***gaitset_submit_bs${bs}.om***</u>模型文件。

2. 开始推理验证。<u>***根据实际推理工具编写***</u>

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
      python -m ais_bench --model=gaitset_submit_bs${bs}.om --input=./CASIA-B-bin --output=./ --output_dirname=./result --batchsize=${batch_size}     
        ```

        -   参数说明：

             -   model：om模型地址
             -   input：预处理数据
             -   output：推理结果保存路径
             -   output_dirname:推理结果保存子目录

        推理后的输出保存在当前目录result下。


   3. 精度验证。

      调用脚本GaitSet_postprocess.py，可以获得Accuracy数据。

      ```
      python GaitSet_postprocess.py --output_path=./result
      ```

      - 参数说明：

        - output_path：推理结果保存地址

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
      python -m ais_bench --model=gaitset_submit_bs${bs}.om --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型路径
        - --batchsize：batchsize大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|     Ascend310P3      |        1          |     GaitDatasetB-silh       |      95.512%      |         606        |
|     Ascend310P3      |        4          |     GaitDatasetB-silh       |            |       696          |
|     Ascend310P3      |       8          |     GaitDatasetB-silh       |            |         703        |
|     Ascend310P3      |       16          |     GaitDatasetB-silh       |            |        714         |
|     Ascend310P3      |        32          |     GaitDatasetB-silh       |            |       720          |
|     Ascend310P3      |        64          |     GaitDatasetB-silh       |            |        723         |