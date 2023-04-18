# RARE_Resnet34_vd模型-推理指导


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

RARE是一个对于不规则的文字具有鲁棒性的识别模型模型，参考论文[[Robust Scene Text Recognition with Automatic Rectification](https://arxiv.org/abs/1603.03915v2)]，它的深度神经网络，包括一个空间变换网络Spatial Transformer Network (STN)和一个序列识别网络Sequence Recognition Network (SRN)，两个网络同时用BP算法进行训练。


- 参考实现：

  ```
  url=https://github.com/PaddlePaddle/PaddleOCR.git
  branch=release/2.5
  commit_id=a40f64a70b8d290b74557a41d869c0f9ce4959d5
  model_name=RARE_Resnet34_vd
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 32 x 100 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 25 x 38 | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                    | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | paddlepaddle                                                 | 2.3.2   | 该依赖只支持x86                                               |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。  | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone -b release/2.5 https://github.com/PaddlePaddle/PaddleOCR.git
   cd PaddleOCR 
   git reset --hard a40f64a70b8d290b74557a41d869c0f9ce4959d5
   git apply ../RARE_Resnet34_vd.patch
   python3 setup.py install
   export PYTHONPATH=$(echo $(pwd)):$PYTHONPATH
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   该模型在以LMDB格式(LMDBDataSet)存储的IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估，共计12067个评估数据，数据介绍参考[DTRB](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)。
   
   下载后将其中的`evaluation.zip`压缩包存放在`RARE_Resnet34_vd`目录下，并通过以下命令进行解压。 

   ```
   mkdir -p ./train_data/data_lmdb_release/
   unzip -d ./train_data/data_lmdb_release/ evaluation.zip
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   在`RARE_Resnet34_vd`工作目录下，执行RARE_Resnet34_vd_preprocess.py脚本，完成预处理。

   ```
   python3 RARE_Resnet34_vd_preprocess.py \
        --config=PaddleOCR/configs/rec/rec_r34_vd_tps_bilstm_att.yml \
        --opt=bin_data=rare_bindata
   ```

   - 参数说明：

      - --config：模型配置文件。
      - --opt=bin_data：bin文件保存路径。

   运行后在当前目录下的`rare_bindata`路径中保存生成的二进制数据。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用`paddle2onnx`将模型权重文件转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       训练权重链接为：https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_att_v2.0_train.tar。
       
       在`RARE_Resnet34_vd`工作目录下可通过以下命令获取训练权重并转为推理模型。

       ```
       wget -nc -P ./checkpoint https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_att_v2.0_train.tar
       cd ./checkpoint && tar xf rec_r34_vd_tps_bilstm_att_v2.0_train.tar && cd ..
       python3 PaddleOCR/tools/export_model.py \
           -c PaddleOCR/configs/rec/rec_r34_vd_tps_bilstm_att.yml \
           -o Global.pretrained_model=./checkpoint/rec_r34_vd_tps_bilstm_att_v2.0_train/best_accuracy  \
           Global.save_inference_dir=./inference/rec_rare

       ```
      
       - 参数说明：

            -   -c：模型配置文件。
            -   -o: 模型入参信息。
            -   Global.pretrained_model：权重文件保存路径。
            -   Global.save_inference_dir：paddleocr推理模型保存路径。

   2. 导出onnx文件。

      1. 使用paddle2onnx工具导出onnx文件。

         在`RARE_Resnet34_vd`工作目录下通过运行以下命令获取onnx模型。

         ```
         paddle2onnx \
             --model_dir ./inference/rec_rare \
             --model_filename inference.pdmodel \
             --params_filename inference.pdiparams \
             --save_file ./RARE_Resnet34_vd_dybs.onnx \
             --opset_version 11 \
             --enable_onnx_checker True
         ```

         参数说明请通过`paddle2onnx -h`命令查看。

         运行后在`RARE_Resnet34_vd`目录下获得RARE_Resnet34_vd_dybs.onnx文件。
    
      2. 优化ONNX文件。
         请访问[auto-optimizer推理工具](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)代码仓，根据readme文档进行工具安装。

         使用opt_onnx.py脚本优化onnx模型，主要是替换GridSample算子。

         ```
          python3 opt_onnx.py \
              --in_onnx=./RARE_Resnet34_vd_dybs.onnx \
              --out_onnx=./RARE_Resnet34_vd_opt_dybs.onnx
         ```

         - 参数说明：

           -   --in_onnx：输入ONNX模型文件。
           -   --out_onnx：输出ONNX模型文件。

         获得RARE_Resnet34_vd_opt_dybs.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

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
         atc --framework=5 \
             --model=./RARE_Resnet34_vd_opt_dybs.onnx \
             --output=./RARE_Resnet34_vd_bs${bs} \
             --input_format=NCHW \
             --input_shape="x:${bs},3,32,100" \
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

           `${bs}`表示om模型可支持不同batch推理，可取值为：1，4，8，16，32，64。
           运行成功后生成`RARE_Resnet34_vd_bs${bs}.om`模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  


   2. 执行推理。

      ```shell
      python3 -m ais_bench \
          --model=./RARE_Resnet34_vd_bs${bs}.om \
          --input=./rare_bindata \
          --output=./ --output_dirname output
      ```

      -   参数说明：

           -   --model：om模型路径。
           -   --input：bin文件路径。
           -   --output：推理结果保存路径。
           -   --output_dirname：推理结果子目录。

      推理完成后在当前目录生成output文件夹。


   3. 精度验证。

      执行后处理脚本`RARE_Resnet34_vd_postprocess.py`，参考命令如下：

      ```
      python3 RARE_Resnet34_vd_postprocess.py \
          --config=PaddleOCR/configs/rec/rec_r34_vd_tps_bilstm_att.yml \
          --opt=results=output
      ```

      -   参数说明：

            -   --config：模型配置文件。
            -   --opt：推理结果路径。


      推理结果通过屏显显示，如下所示：

      ```
      {'acc': 0.847932376855944, 'norm_edit_dis': 0.9322851725751706}
      ```

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```shell
      python3 -m ais_bench \
          --model=./RARE_Resnet34_vd_bs${bs}.om \
          --loop=50 \
          --batchsize=${bs}
      ```

      -   参数说明：

          -   --model：om模型路径。
          -   --loop：推理次数。
          -   --batchsize：om模型的batch。

      `${bs}`表示不同batch的om模型。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|Ascend310P3| 1                | IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE | {'acc': 0.8479, 'norm_edit_dis': 0.9322} | 296.48 |
|Ascend310P3| 4                | IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE | {'acc': 0.8479, 'norm_edit_dis': 0.9322} | 949.78 |
|Ascend310P3| 8                | IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE | {'acc': 0.8479, 'norm_edit_dis': 0.9322} | 1161.38 |
|Ascend310P3| 16               | IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE | {'acc': 0.8479, 'norm_edit_dis': 0.9322} | 1431.58 |
|Ascend310P3| 32               | IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE | {'acc': 0.8479, 'norm_edit_dis': 0.9322} | 1603.60 |
|Ascend310P3| 64               | IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE | {'acc': 0.8479, 'norm_edit_dis': 0.9322} | 1601.73 |