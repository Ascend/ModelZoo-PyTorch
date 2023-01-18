# ch_ppocr_mobile_v2.0_cls模型-推理指导


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
ch_ppocr_mobile_v2.0_cls为[[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/angle_class.md#%E6%96%B9%E6%B3%95%E4%BB%8B%E7%BB%8D)]内置的中文文本方向分类器，对检测到的文本行文字角度分类，支持0和180度的分类。文本方向分类器主要用于图片非0度的场景下，该场景下需要对图片里检测到的文字进行分类，便于后续进行转正操作后将图片送入识别模型。



- 参考实现：

  ```
  url=https://github.com/PaddlePaddle/PaddleOCR
  commit_id=a40f64a70b8d290b74557a41d869c0f9ce4959d5
  model_name=ch_ppocr_mobile_v2.0_cls
  ```





## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 48 x 192 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 2 | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | paddlepaddle                                                 | 2.3.2   | 仅支持X86服务器安装                                           |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone -b release/2.5 https://github.com/PaddlePaddle/PaddleOCR.git
   cd PaddleOCR
   git reset --hard a40f64a70b8d290b74557a41d869c0f9ce4959d5
   git apply ../ch_ppocr_mobile_v2.0_cls.patch
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   cd PaddleOCR
   python3 setup.py install
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   该模型在PaddleOCR提供的中文文本方向分类[[样本集](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.5/doc/imgs_words/ch)]进行精度验证，该样本集在 ./PaddleOCR/doc/imgs_words/ch 目录下。目录结构如下：

   ```
   ch_ppocr_mobile_v2.0_cls
   ├── PaddleOCR
      ├── doc
          ├── imgs_words
              ├── ch
                  ├── words_1.jpg
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行ch_ppocr_mobile_v2.0_cls_preprocess.py脚本，完成预处理。

   ```
   python3 ch_ppocr_mobile_v2.0_cls_preprocess.py \
        -c PaddleOCR/configs/cls/cls_mv3.yml \
        -o Global.infer_img=PaddleOCR/doc/imgs_words/ch
   ```
   运行成功后，在pre_data文件夹下生成bin文件。目录结构如下：
    ```
   ch_ppocr_mobile_v2.0_cls
   ├── PaddleOCR
   ├── pre_data
      ├── words_1.bin
    ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用paddle2onnx工具将模型权重文件转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

        推理权重下载链接为：https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar。

        通过以下命令可以得到权重文件。

       ```
       wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
       cd ./inference
       tar xf ch_ppocr_mobile_v2.0_cls_infer.tar
       cd ..
       ```

   2. 导出onnx文件。

      使用paddle2onnx工具导出onnx文件。

        ```
        paddle2onnx \
            --model_dir ./inference/ch_ppocr_mobile_v2.0_cls_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ./ch_ppocr_mobile_v2.0_cls.onnx \
            --opset_version 11 \
            --enable_onnx_checker True \
            --input_shape_dict="{'x':[-1,3,48,192]}"
        ```

        获得ch_ppocr_mobile_v2.0_cls.onnx文件。

        参数说明请通过`paddle2onnx -h`命令查看

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
             --model=./ch_ppocr_mobile_v2.0_cls.onnx \
             --output=./ch_ppocr_mobile_v2.0_cls_bs${batchsize} \
             --input_format=NCHW \
             --input_shape="x:${batchsize},3,48,192" \
             --log=error \
             --soc_version=Ascend${chip_name} \
             --insert_op_conf=./aipp_ch_ppocr_mobile_v2.0_cls.config \
             --enable_small_channel=1
         ```

         - 参数说明：

            -   --model：为ONNX模型文件。
            -   --framework：5代表ONNX模型。
            -   --output：输出的OM模型。
            -   --input\_format：输入数据的格式。
            -   --input\_shape：输入数据的shape。
            -   --log：日志级别。
            -   --soc\_version：处理器型号。
            -   --insert_op_conf：插入算子的配置文件路径与文件名，例如aipp预处理算子。
            -   --enable_small_channel：是否使能small channel的优化，使能后在channel<=4的首层卷积会有性能收益。

           运行成功后生成`ch_ppocr_mobile_v2.0_cls_bs${batchsize}`模型文件。`${batchsize}`为模型输入的batch size。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3 -m ais_bench \
        --model=./ch_ppocr_mobile_v2.0_cls_bs${batchsize}.om \
        --input=./pre_data \
        --output=./ \
        --batchsize=${batchsize}
        ```

        -   参数说明：

            -   --model：om模型路径。
            -   --input：bin文件路径。
            -   --output：推理结果保存路径。
            -   --batchsize：模型输入的batch size。

        推理后的输出默认在当前目录下。


   3. 精度验证。

      调用脚本得到数据集的推理结果，结果通过屏显显示。

      ```
      python3 ch_ppocr_mobile_v2.0_cls_postprocess.py --config=PaddleOCR/configs/cls/cls_mv3.yml --opt=results=${output_path}
      ```

      - 参数说明：

        - config：模型配置文件。
        - opt：模型推理保存路径。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=$ch_ppocr_mobile_v2.0_cls_bs${batchsize}.om --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型文件路径。
        - --batchsize：模型输入对应的batch size。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|    310P       |        1     |    样例图片    |     与在线推理一致       |     2607.56      |
|    310P       |        4     |           |            |     8621.80     |
|    310P       |       8      |           |            |     14915.91    |
|    310P       |       16     |           |            |     21204.41    |
|    310P       |       32     |           |            |     26886.69    |
|    310P       |       64     |    样例图片    |      与在线推理一致      |     30269.34    |