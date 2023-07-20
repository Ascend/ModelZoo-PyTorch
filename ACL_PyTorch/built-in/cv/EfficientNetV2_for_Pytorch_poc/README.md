# EfficientNetV2模型-推理指导


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

EfficientNetV2是一系列图像分类模型，与现有技术相比，其实现了更好的参数效率和更快的训练速度。基于EfficientNetV1，Efficient NetV2模型使用神经架构搜索（NAS）来联合优化模型大小和训练速度，并以更快的训练和推理速度进行扩展。



- 参考实现：

  ```
  url=https://github.com/rwightman/pytorch-image-models
  ```





## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 288 x 288 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 1000 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动 

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.11.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型使用ImageNet官网的5万张验证集进行测试。用户需自行获取数据集（或给出明确下载链接），将数据集解压并上传到源码包路径下。目录结构如下：

   ```
   val
   ├── n02071294       
      ├── ILSVRC2012_val_00010966.JPEG       
      ├── ...       
   ├── n04532670       
   ├── ...
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行preprocess.py脚本，完成预处理。

   ```
   python3 preprocess.py --dataset_path=./val --save_path=./bin_data --aipp_save_path=./aipp_bin_data
   ```

   - 参数说明：

      -   --dataset_path：数据集路径。
      -   --save_path：预处理后bin文件保存路径，用于模型量化，默认为./bin_data。
      -   --aipp_save_path：适配AIPP的预处理后bin文件保存路径，用于实际推理，默认为./aipp_bin_data。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       执行pth2onnx.py时自动下载

   2. 导出onnx文件。

      1. 使用pth2onnx.py导出onnx文件。

         ```
         python3 pth2onnx.py
         ```

         获得efficientnetv2.onnx文件。

      2. 模型量化

         请访问[昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/devtools/auxiliarydevtool/atlasamctonnx_16_0012.html)，跟据安装指南安装amct_onnx工具。

         生成校准数据，进行模型量化
      
         ```
         amct_onnx calibration --model efficientnetv2.onnx --save_path amct_model/efficientnetv2 --input_shape "image:1,3,288,288" --data_dir ./bin_data --data_types "float32" --calibration_config quant.cfg
         ```

         - 参数说明：

            -   --model：原始onnx模型。
            -   --save_path：量化后文件保存路径。
            -   --input_shape：模型输入shape。
            -   --data_dir：用于模型量化的数据路径。
            -   --data_types: 用于量化数据的类型。
            -   --calibration_config：量化配置文件。
            
         在amct_model文件夹下得到efficientnetv2_deploy_model.onnx。

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
         atc --model=amct_model/efficientnetv2_deploy_model.onnx --framework=5 --input_format=NCHW --input_shape="image:24,3,288,288" --output=efficientnetv2_bs24 --soc_version=Ascend${chip_name} --log=error --optypelist_for_implmode="Sigmoid" --op_select_implmode=high_performance --insert_op_conf=aipp.cfg --enable_small_channel=1
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --insert\_op\_conf:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。

           运行成功后生成<u>***efficientnetv2_bs24.om***</u>模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
   
   3. 执行推理。

        ```
        python3 -m ais_bench --model=efficientnetv2_bs24.om --input=aipp_bin_data --output=./ --output_dir result --outfmt BIN --device 0
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --input：预处理后二进制目录。
             -   --output：推理结果输出路径。
             -   --output_dir: 推理结果输出目录名。
             -   --outfmt：推理结果输出格式。
             -   --device: 推理使用的device id。

        推理后的输出默认在当前目录result下。


   4. 精度验证。

      调用脚本与数据集标签label.txt比对，可以获得Accuracy数据。

      ```
      python3 postprocess.py --output_dir=result --label_path=val_label.txt
      ```

      - 参数说明：

        - --output_dir：为生成推理结果所在路径。
        - --label_path：数据集标签文件。
  

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
|   310P3        |      24            |    ImageNet        |     81%       |      2144fps           |