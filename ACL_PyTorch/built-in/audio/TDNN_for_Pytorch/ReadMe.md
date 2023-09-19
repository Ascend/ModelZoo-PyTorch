# TDNN模型-推理指导


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

TDNN是一种经典的语音识别网络结构，主要由Conv1D+Relu+BN组成，speechbrain在原有结构上添加了最新的压缩网络和激励网络，使得TDNN可以适应更为复杂的任务并且获得更高的精度。


- 参考实现：

  ```
  url=https://github.com/speechbrain/speechbrain.git
  commit_id=51a2becdcf3a337578a9307a0b2fc3906bf20391
  code_path=speechbrain/templates/speaker_id
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32 | batchsize x 1800 x 24 | ND         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | 1 x 28 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动 

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.10.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/speechbrain/speechbrain.git
   cd speechbrain    
   git checkout  develop    
   git reset --hard 51a2becdcf3a337578a9307a0b2fc3906bf20391
   git apply --reject --whitespace=fix ../modify.patch
   cd ..
   ```
2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
   cd MagicONNX && git checkout 8d62ae9dde478f35bece4b3d04eef573448411c9
   pip3 install .
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型使用Mini Librispeech数据集验证，预处理阶段会自动下载数据集。

2. 获取权重文件。

   [classifier.ckpt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/TDNN/pth/classifier.ckpt) 和 [embedding_model.ckpt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/TDNN/pth/embedding_model.ckpt)  
   将模型权重文件放入当前目录下，将hyperparams.yaml文件放入best_model中
   ```
   mkdir best_model
   mv hyperparams.yaml best_model
   ```

3. 数据预处理，将原始数据集转换为模型输入的数据。

   执行tdnn_preprocess.py脚本，完成预处理。

   ```
   export PYTHONPATH=${PWD}/speechbrain:${PYTHONPATH}
   python3 tdnn_preprocess.py 
   ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 导出onnx文件。

      1. 使用tdnn_pth2onnx.py导出onnx文件。

         ```
         python3 tdnn_pth2onnx.py 64
         ```

         获得tdnn_bs64.onnx文件。

      2. 优化ONNX文件。

         ```
         python3 -m onnxsim tdnn_bs64.onnx tdnn_bs64s.onnx
         python3 modify_onnx.py tdnn_bs64s.onnx
         ```

         获得tdnn_bs64s.onnx文件。

   2. 使用ATC工具将ONNX模型转OM模型。

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
         atc --model=tdnn_bs64s.onnx --framework=5 --input_format=ND --input_shape="feats:64,-1,23" --dynamic_dims='200;300;400;500;600;700;800;900;1000;1100;1200;1300;1400;1500;1600;1700;1800' --output=./tdnn_bs64s --soc_version=Ascend${chip_name} --log=error 
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***tdnn_bs64s.om***</u>模型文件。

2. 开始推理验证。

   1. 执行推理。

        ```
        python3 tdnn_pyacl_infer.py --model_path=tdnn_bs64s.om --batch_size=64 --device_id=0 --cpu_run=True --sync_infer=True --input_info_file_path=mini_librispeech_test.info --input_dtypes=float32 --infer_res_save_path=./result --res_save_type=bin  
        ```

        -   参数说明：

             -   --model_path：om文件路径。
             -   --batch_size：模型batch size。
             -   --device_id：NPU设备编号。
             -   --cpu_run：MeasureTime类的cpu_run参数，True or False。
             -   --sync_infer：同步推理，True or False。
             -   --input_info_file_path：预处理时生成的数据集info文件。
             -   --input_dtypes：输入数据类型。
             -   --infer_res_save_path：推理输出保存目录
             -   --res_save_type：推理输出保存格式
             

        推理后的输出默认在当前目录result下。


   2. 精度验证。

      调用脚本与数据集标签比对，可以获得Accuracy数据。

      ```
      python3 tdnn_postprocess.py
      ```

   3. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size和length的om模型的性能。  
	  安装ais_bench推理工具。请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  
	  推理参考命令如下：

        ```
         python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size} --dymDims=feats:${batch_size},${length},23
        ```

      - 参数说明：
        - --model：om模型路径
        - --batchsize：模型batch大小
        - --dymDims：推理输入的实际shape



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | length | 数据集 | 精度 | 性能 |
| --------- | -------- | ---------- | ---------- | ---------- | --------------- |
|    Ascend310P       |    1   |   1800    |   Mini Librispeech         |     99.93%       |      465.21fps           |
|    Ascend310P       |    4   |   1800    |   Mini Librispeech         |     99.93%       |      1415.17fps           |
|    Ascend310P       |    8   |   1800    |   Mini Librispeech         |     99.93%       |      1058.37fps           |
|    Ascend310P       |    16   |   1800    |   Mini Librispeech         |     99.93%       |      1066.47fps           |
|    Ascend310P       |    32   |   1800    |   Mini Librispeech         |     99.93%       |      1080.21fps           |
|    Ascend310P       |    64   |   1800    |   Mini Librispeech         |     99.93%       |      1682.2fps           |