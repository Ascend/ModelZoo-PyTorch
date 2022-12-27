# TransPose模型-推理指导


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

TransPose是一种基于CNN特征提取器、Transformer编码器和预测头的人体姿态估计模型。给定一幅图像，Transformer中内置的注意力层可以有效地捕捉关键点之间的长距离空间关系，并解释预测的关键点位置高度依赖的依赖性。


- 参考实现：

  ```
  url=https://github.com/yangsenius/TransPose.git
  commit_id=dab9007b6f61c9c8dce04d61669a04922bbcd148
  model_name=TransPose-R-A3
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
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/yangsenius/TransPose.git
   cd TransPose
   git reset dab9007b6f61c9c8dce04d61669a04922bbcd148 --hard
   patch -p1 < ../TransPose.patch
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持COCO2017验证集。用户需自行获取数据集（或给出明确下载链接），将文件夹解压并上传数据集到源码包路径下。目录结构如下：

   ```
   |-- data
        |-- coco
            |-- images
            |   |-- val2017
            |       |-- 000000000139.jpg
            |       |-- 000000000285.jpg
            |       |-- 000000000632.jpg
            |       |-- ... 
            |-- annotations
                |-- person_keypoints_train2017.json
                |-- person_keypoints_val2017.json
   ```

   下载与解压，请参考： 
    ```
    mkdir -p data/coco/
    cd data/coco/
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    mkdir images
    unzip val2017.zip -d images
    unzip annotations_trainval2017.zip
    cd ../../
    ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   ```
   python3 TransPose_preprocess.py --output ./prep_data --output_flip ./prep_data_flip
   ```
   参数说明：
   - --output：输出的二进制文件（.bin）所在路径。
   - --output_flip：输出的二进制文件flip（.bin）所在路径。

   运行成功后，会在当前目录下生成 prep_data 与 prep_data_flip 目录，用于保存生成的bin文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      获取权重文件“tp_r_256x192_enc3_d256_h1024_mh8.pth”，将文件放入models文件夹内。
      ```
      mkdir models
      wget https://github.com/yangsenius/TransPose/releases/download/Hub/tp_r_256x192_enc3_d256_h1024_mh8.pth -P models
      ```

   2. 导出onnx文件。

      1. 使用TransPose_pth2onnx.py导出onnx文件。

         ```
         python3 TransPose_pth2onnx.py --weights models/tp_r_256x192_enc3_d256_h1024_mh8.pth
         ```
         获得tp_r_256x192_enc3_d256_h1024_mh8.onnx文件。 

      2. 优化ONNX文件。

         ```
         python3 -m onnxsim models/tp_r_256x192_enc3_d256_h1024_mh8.onnx models/tp_r_256x192_enc3_d256_h1024_mh8_sim.onnx --dynamic-input-shape
         ```
         获得tp_r_256x192_enc3_d256_h1024_mh8_sim.onnx文件。

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
         atc --framework=5 --model=models/tp_r_256x192_enc3_d256_h1024_mh8_sim.onnx --output=models/tp_r_256x192_enc3_d256_h1024_mh8_bs1 --input_format=NCHW --input_shape="input:1,3,256,192" --fusion_switch_file=fusion_switch.cfg --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***tp_r_256x192_enc3_d256_h1024_mh8_bs1.om***</u>模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
        python3 -m ais_bench --model models/tp_r_256x192_enc3_d256_h1024_mh8_bs{batchsize}.om --input prep_data/ --output prep_data_result/
        python3 -m ais_bench --model models/tp_r_256x192_enc3_d256_h1024_mh8_bs{batchsize}.om --input prep_data_flip --output prep_data_flip_result/  
        ```

        -   参数说明：

            - --model：OM模型路径
            - --input：存放预处理 bin 文件的目录路径
            - --output：推理输出文件夹


   3. 精度验证。

      调用脚本与数据集标签比对，可以获得Accuracy数据。

      ```
      python3 TransPose_postprocess.py  --dump_dir ./prep_data_result/${output} --dump_dir_flip ./prep_data_flip_result/${output}
      ```

      - 参数说明：

         - --dump_dir：生成推理结果所在路径。
         - --dump_dir_flip：生成推理结果所在路径。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型路径
        - --batchsize：batch大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|   310P3        |       1           |   COCO2017         |    AP: 73.7%        |      484.04           |
|   310P3        |       4           |   COCO2017         |    AP: 73.7%        |      500.14           |
|   310P3        |       8           |   COCO2017         |    AP: 73.7%        |      485.39           |
|   310P3        |       16           |   COCO2017         |    AP: 73.7%        |      458.66           |
|   310P3        |       32           |   COCO2017         |    AP: 73.7%        |      437.97           |
|   310P3        |       64           |   COCO2017         |    AP: 73.7%        |      475.99           |