# PSENet_ResNet50_vd模型-推理指导

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

PSENet([Shape Robust Text Detection with Progressive Scale Expansion Network](https://arxiv.org/abs/1903.12473))是一种新颖的渐进式扩展网络，它可以精确地检测具有任意形状的文本实例。进一步来说，PSENet 为每个文本实例生成不同尺度的内核，并逐渐将最小尺度内核扩展到具有完整形状的文本实例。由于最小尺度内核之间存在较大的几何边距，该方法可以有效地分割紧密的文本实例，从而更容易使用基于分割的方法来检测任意形状的文本实例。

- 参考实现：

  ```
  url=https://github.com/PaddlePaddle/PaddleOCR.git
  branch=release/2.5
  commit_id=a40f64a70b8d290b74557a41d869c0f9ce4959d5
  model_name=PSENet_ResNet50_vd
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 736 x 1312 | NCHW        |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 7 x 184 x 328 | NCHW         |


# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| paddlepaddle                                                 | 2.3.2   | 仅支持x86环境安装                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone -b release/2.5 https://github.com/PaddlePaddle/PaddleOCR.git
   cd PaddleOCR 
   git reset --hard a40f64a70b8d290b74557a41d869c0f9ce4959d5
   git apply ../PSENet_ResNet50_vd.patch
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   cd PaddleOCR
   python3 setup.py install
   export PYTHONPATH=$(echo $(pwd)):$PYTHONPATH
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   IICDAR 2015 数据集包含1000张训练图像和500张测试图像。参考PaddleOCR数据集数据处理方式，ICDAR 2015 数据集可以点击链接进行下载，本模型需下载Test Set Images(43.3MB)。

   将数据集`ch4_test_images.zip`放在工作目录下，通过以下命令创建`train_data/icdar2015/ch4_test_images`路径，并通过以下命令进行解压保存并获取标签文件。
   ```
   mkdir -p ./train_data/icdar2015/ch4_test_images/
   unzip -d ./train_data/icdar2015/ch4_test_images/ ch4_test_images.zip
   wget -P ./train_data/icdar2015/ https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   在`PSENet_ResNet50_vd`工作目录下，执行PSENet_ResNet50_vd_preprocess.py脚本，完成预处理。

   ```
    python3 PSENet_ResNet50_vd_preprocess.py \
        -c PaddleOCR/configs/det/det_r50_vd_pse.yml \
        -o data_dir=./train_data/icdar2015/ bin_dir=./data_bin info_dir=./data_info
   ```

   - 参数说明：

       -   -c：模型配置文件。
       -   -o：配置参数，其中data_dir表示数据集路径，bin_dir表示二进制数据保存路径，data_info数据信息保存路径。

   运行后在当前目录下的`data_bin`路径中保存生成的二进制数据，`data_info`路径中保存数据信息。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用`paddle2onnx`将模型权重文件转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      训练权重链接为：https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_vd_pse_v2.0_train.tar。

      在`PSENet_ResNet50_vd`工作目录下可通过以下命令获取训练权重并转为推理模型。
      ```
      wget -nc -P ./checkpoint https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_vd_pse_v2.0_train.tar
      cd ./checkpoint && tar xf det_r50_vd_pse_v2.0_train.tar && cd ..
      python3 PaddleOCR/tools/export_model.py \
           -c PaddleOCR/configs/det/det_r50_vd_pse.yml \
           -o Global.pretrained_model=./checkpoint/det_r50_vd_pse_v2.0_train/best_accuracy  \
           Global.save_inference_dir=./inference/det_pse
      ```
      
      - 参数说明：

         -  -c：模型配置文件。
         -  -o: 模型入参信息。
         -  Global.pretrained_model：权重文件保存路径。
         -  Global.save_inference_dir：paddleocr推理模型保存路径。

   2. 导出onnx文件。

      1. 使用paddle2onnx工具导出onnx文件。

         运行paddle2onnx工具获取onnx模型。

         ```
         paddle2onnx \
             --model_dir ./inference/det_pse \
             --model_filename inference.pdmodel \
             --params_filename inference.pdiparams \
             --save_file ./PSENet_ResNet50_vd_dybs.onnx \
             --opset_version 11 \
             --enable_onnx_checker True \
             --input_shape_dict="{'x':[-1,3,736,1312]}"
         ```

         参数说明请通过`paddle2onnx -h`命令查看。
         运行后获得PSENet_ResNet50_vd_dybs.onnx文件。

      2. 优化onnx模型。
         请访问[auto-optimizer优化工具](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)代码仓，根据readme文档进行工具安装。

         运行modify_onnx.py脚本优化onnx模型，优化点为：Resize算子按Paddle定义参数导出的onnx模型有精度问题，因此将按PyTorch定义重新构造Resize参数。

         ```
         python3 modify_onnx.py PSENet_ResNet50_vd_dybs.onnx PSENet_ResNet50_vd_md_dybs.onnx
         ```

         参数说明:第一个为输入onnx，第二个为输出onnx。

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
             --model=./PSENet_ResNet50_vd_sim_md_dybs.onnx \
             --output=./PSENet_ResNet50_vd_bs${bs} \
             --input_format=NCHW \
             --input_shape="x:${bs},3,736,1312" \
             --log=error \
             --soc_version=Ascend${chip_name} \
             --insert_op_conf=PSENet_ResNet50_vd_aipp.cfg
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

           运行成功后生成`PSENet_ResNet50_vd_bs${bs}.om`模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  


   2. 执行推理。

      ```
      python3 -m ais_bench \
          --model=./PSENet_ResNet50_vd_bs${bs}.om \
          --input=./data_bin \
          --output=./ \
          --output_dirname=output
      ```

      -  参数说明：

         -  --model：om模型路径。
         -  --input：bin文件路径。
         -  --output：推理结果保存路径。
         -  --output_dirname：推理结果保存路径子目录。

      推理完成后在当前目录生成output文件夹。


   3. 精度验证。

      执行后处理脚本`PSENet_ResNet50_vd_postprocess.py`，参考命令如下：

      ```
      python3 PSENet_ResNet50_vd_postprocess.py \
        -c PaddleOCR/configs/det/det_r50_vd_pse.yml \
        -o info_dir=./data_info res_dir=./output
      ```

      -  参数说明：

         - -c：模型配置文件。
         - -o：可选参数，info_dir表示数据信息路径；res_dir表示推理结果路径。

      推理结果通过屏显显示。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model=./PSENet_ResNet50_vd_bs${bs}.om --loop=50
      ```

      -  参数说明：

         - --model：om模型路径。
         - --loop：推理次数。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ----------- | ---------- | ---------- | --------------- |
|Ascend310P3| 1           | ICDAR 2015 | precision: 0.8572 |  51.138  |
|Ascend310P3| 4           | ICDAR 2015 | precision: 0.8572 |  47.276  |
|Ascend310P3| 8           | ICDAR 2015 | precision: 0.8572 |  47.675  |
|Ascend310P3| 16          | ICDAR 2015 | precision: 0.8572 |  48.256  |
|Ascend310P3| 32          | ICDAR 2015 | precision: 0.8572 |  48.538  |
|Ascend310P3| 64          | ICDAR 2015 | precision: 0.8572 |  38.610  |