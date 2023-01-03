# Uniformer模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

UniFormer 提出了一种整合 3D 卷积和时空自注意力机制的 Transformer 结构，能在计算量和精度之间取得平衡。不同于传统的 Transformer 结构在所有层都使用自注意力机制，论文中提出的 relation aggregator 可以分别处理视频的冗余信息和依赖信息。在浅层，aggregator 利用一个小的 learnable matrix 学习局部的关系，通过聚合小的 3D 邻域的 token 信息极大地减少计算量。在深层，aggregator通过相似性比较学习全局关系，可以灵活的建立远距离视频帧 token 之间的长程依赖关系。



- 参考实现：

  ```
  url=https://github.com/Sense-X/UniFormer
  commit_id=e8024703bffb89cb7c7d09e0d774a0d2a9f96c25
  model_name=UniFormer
  ```






## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 256 x 192 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 17 x 64 x 48 | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动 

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone -b main https://github.com/Sense-X/UniFormer.git
   cd UniFormer
   git reset e8024703bffb89cb7c7d09e0d774a0d2a9f96c25 --hard
   cd pose_estimation
   python setup.py install 
   cd ..
   patch -p1 < ../uniformer.patch
   cd ..
   ```

2. 安装依赖。

   ```
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）


   本模型支持coco验证集。用户需自行获取数据集（或给出明确下载链接），将.json文件和val2017文件夹解压并上传数据集到源码包路径下。目录结构如下：

   ```
   coco
   ├── annotations
   │   └─person_keypoints_val2017.json    //验证集标注信息       
   └── val2017             // 验证集文件夹
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行uniformer_preprocess.py脚本，完成预处理。

   ```
   python uniformer_preprocess.py --dataset=./coco --bin=pre_data
   ```
      -   参数说明：

            -   dataset：数据集目录
            -   bin：预处理数据保存地址



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      ```
      wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Uniformer/PTH/top_down_256x192_global_base.pth
      ```

   2. 导出onnx文件。

      1. 使用源码库自带脚本导出onnx文件。

         运行pytorch2onnx.py脚本。

         ```
         python UniFormer/pose_estimation/tools/pytorch2onnx.py \
                UniFormer/pose_estimation/exp/top_down_256x192_global_base/config.py \
               ./top_down_256x192_global_base.pth \
               --output-file ./uniformer_dybs.onnx
         ```

         获得uniformer_dybs.onnx文件。

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
             --model=uniformer_dybs.onnx \
             --output=uniformer_bs${bs} \
             --input_format=NCHW \
             --input_shape="input:${bs},3,256,192" \
             --log=error \
             --soc_version=${soc_version}  
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***uniformer_bs${bs}.om***</u>模型文件。

2. 开始推理验证。

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
      python ${ais_infer_path}/ais_infer.py --model=uniformer_bs${bs}.om --input=./pre_data --output=./ --output_dirname=./result --batchsize=${batch_size}  
        ```

        -   参数说明：

             -   model：om模型地址
             -   input：预处理数据
             -   output：推理结果保存路径
             -   output_dirname:推理结果保存子目录

        推理后的输出保存在当前目录result下。

        >**说明：** 
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   3. 精度验证。

      调用脚本uniformer_postprocess.py

      ```
      python uniformer_postprocess.py --dataset=coco --bin=result
      ```

      - 参数说明：

        - bin：为生成推理结果所在路径


        - dataset：数据集目录

   4. 性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
      python ${ais_infer_path}/ais_infer.py --model=uniformer_bs${bs}.om --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型路径
        - --batchsize：batchsize大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|    Ascend310P3       |        1          |      coco      |     93.5       |       178.8          |
|    Ascend310P3       |       4          |      coco      |            |          210       |
|    Ascend310P3       |        8          |      coco      |            |      295           |
|    Ascend310P3       |        16          |      coco      |            |       286          |
|    Ascend310P3       |        32          |      coco      |            |       257          |
|    Ascend310P3       |        64         |      coco      |            |       243          |