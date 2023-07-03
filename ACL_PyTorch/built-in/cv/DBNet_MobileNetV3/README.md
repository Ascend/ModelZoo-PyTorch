# DBNet_MobileNetV3模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

DBNet([Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion](https://arxiv.org/abs/2202.10304))一个名为可微分二值化（DB）的模块，它可以在分割网络中执行二值化过程。与DB模块一起优化的分割网络可以自适应地设置用于二值化的阈值，这不仅简化了后处理，还提高了文本检测的性能。基于一个简单的分割网络，我们在五个基准数据集上验证了DB的性能改进，这在检测精度和速度方面始终达到了最先进的结果。特别是，对于轻量级主干，DB的性能改进非常显著，因此我们可以在检测精度和效率之间寻找理想的折衷方案。

- 参考实现：

  ```
  url=https://github.com/PaddlePaddle/PaddleOCR.git
  branch=release/2.6
  model_name=DBNet_MobileNetV3
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | x    | UINT8 | batchsize x 736 x 1280 x 3  | NHWC         |


- 输出数据

  | 输出数据 | 数据类型     | 大小 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | sigmoid_0.tmp_0  | FLOAT32 | batchsize x 1 x 736 x 1280  | NCHW           |


# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.4  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| paddlepaddle                                                 | 2.3.2   | 仅支持x86环境安装                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git
   cd PaddleOCR 
   patch -p1 < ../DBNet_MobileNetV3.patch
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   cd PaddleOCR
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   ICDAR 2015 数据集包含1000张训练图像和500张测试图像。参考[[PaddleOCR数据集](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/dataset/ocr_datasets.md)]数据处理方式，ICDAR 2015 数据集可以点击[[链接](https://rrc.cvc.uab.es/?ch=4&com=downloads)]进行下载，首次下载需注册。注册完成登陆后，点击[链接](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=4&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy8/Y29tPWRvd25sb2FkcyZhY3Rpb249ZG93bmxvYWQmZmlsZT1jaDRfdGVzdF9pbWFnZXMuemlw)下载数据集。

   将数据集`ch4_test_images.zip`放在`DBNet_MobileNetV3`工作目录下，通过以下命令创建`PaddleOCR/test_data/icdar2015/text_localization/`路径，将下载的数据集保存该路径下，并在该路径下通过以下命令进行解压保存并获取标签文件。
   ```
   mkdir -p PaddleOCR/test_data/icdar2015/text_localization/ch4_test_images/
   unzip -d PaddleOCR/test_data/icdar2015/text_localization/ch4_test_images/ ch4_test_images.zip
   wget -P PaddleOCR/test_data/icdar2015/text_localization/ https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt
   ```

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   在`DBNet_MobileNetV3`工作目录下，执行DBNet_MobileNetV3_preprocess.py脚本，完成预处理。

   ```
    python3 DBNet_MobileNetV3_preprocess.py \
        -c PaddleOCR/configs/det/det_mv3_db.yml \
        -o data_dir=./PaddleOCR/test_data/icdar2015/text_localization/ bin_dir=./data_bin info_dir=./data_info
   ```

   - 参数说明：

       -   -c：模型配置文件。
       -   -o：配置参数，其中data_dir表示数据集路径，bin_dir表示二进制数据保存路径，data_info数据信息保存路径。

   运行后在当前目录下的`data_bin`路径中保存生成的二进制数据，`data_info`路径中保存数据信息。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用`paddle2onnx`将模型权重文件转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       训练权重链接为：https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar。
       在`DBNet_MobileNetV3`工作目录下可通过以下命令获取训练权重并转为推理模型。
       ```
       wget -nc -P ./checkpoint https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar
       cd ./checkpoint && tar xf det_mv3_db_v2.0_train.tar && cd ..
       python3 PaddleOCR/tools/export_model.py \
           -c PaddleOCR/configs/det/det_mv3_db.yml \
           -o Global.pretrained_model=./checkpoint/det_mv3_db_v2.0_train/best_accuracy  \
           Global.save_inference_dir=./inference/det_db
       ```
      
       - 参数说明：

            -   -c：模型配置文件。
            -   -o: 模型入参信息。
            -   Global.pretrained_model：权重文件保存路径。
            -   Global.save_inference_dir：paddleocr推理模型保存路径。

   2. 导出onnx文件。
      1. 使用paddle2onnx工具导出onnx文件。

         在DBNet_MobileNetV3工作目录下通过运行以下命令获取onnx模型。

         paddle2onnx \
            --model_dir ./inference/det_db \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ./inference/det_db_onnx/DBNet_MobileNetV3_bs${batchsize}.onnx \
            --opset_version 10 \
            --input_shape_dict="{'x':[${batchsize},3,-1,-1]}" \
            --enable_onnx_checker True

         参数说明请通过paddle2onnx -h命令查看。 运行后在DBNet_MobileNetV3/inference/det_db_onnx目录下获得DBNet_MobileNetV3_bs${batchsize}.onnx文件。
         `${batchsize}`表示onnx模型可支持不同batch推理，可取值为：1，4，8，16，32，64。

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
         +------------------------------------------------------------------------------------------------+
         | npu-smi 22.0.2                           Version: 22.0.4                                       |
         +-----------------------+-----------------+------------------------------------------------------+
         | NPU     Name          | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device        | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +=======================+=================+======================================================+
         | 2144    310P3         | OK              | NA           47                0    / 0              |
         | 0       0             | 0000:86:00.0    | 0            1807 / 21527                            |
         +=======================+=================+======================================================+

         ```

      3. 执行ATC命令。
	  
         ```
         atc --framework=5 \
         --model=./inference/det_db_onnx/DBNet_MobileNetV3_bs${batchsize}.onnx \
         --output=./inference/det_db_om/DBNet_MobileNetV3_bs${batchsize} \
         --input_format=NCHW \
         --input_shape="x:${batchsize},3,736,1280" \
         --log=debug \
         --soc_version=Ascend${chip_name} \
         --insert_op_conf=DBNet_MobileNetV3_aipp.cfg \
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
           -   --input_format：输入数据的格式。
           -   --enable_small_channel:是否使能small_channel优化。

           `${batchsize}`表示om模型可支持不同batch推理，可取值为：1，4，8，16，32，64。
           运行成功后生成`DBNet_MobileNetV3_bs${batchsize}.om`模型文件。

2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  


   b.  执行推理。

      ```
      python3 -m ais_bench \
          --model=./inference/det_db_om/DBNet_MobileNetV3_bs${batchsize}.om \
          --input=./data_bin \
          --batchsize=${batchsize} \
          --output=./inference_output

      ```

      -   参数说明：

           -   --model：om模型路径。
           -   --input：bin文件路径。
           -   --batchsize：om模型的batch。
           -   --output：推理结果保存路径。

      `${batchsize}`表示不同batch的om模型。

      推理完成后在当前`DBNet_MobileNetV3`工作目录生成推理结果。其目录命名格式为`xxxx_xx_xx-xx_xx_xx`(`年_月_日-时_分_秒`)，如`2022_08_18-06_55_19`。
      > **说明：** 
      >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。


   c.  精度验证。

      执行后处理脚本`DBNet_MobileNetV3_postprocess.py`，参考命令如下：

      ```
      python3 DBNet_MobileNetV3_postprocess.py \
        -c PaddleOCR/configs/det/det_mv3_db.yml \
        -o info_dir=./data_info res_dir=./inference_output/${output_path}

      ```

      -   参数说明：

            -   -c：模型配置文件。
            -   -o：可选参数，info_dir表示数据信息路径；res_dir表示推理结果路径；${output_path}为om模型推理结果的保存目录，即为b步骤中的output输出目录（年_月_日-时_分_秒）。

      推理结果通过屏显显示，参考如下：

      ```
      precision:0.775
      recall:0.7313432835820896
      hmean:0.7525390141193956

      ```

   d.  性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench \
          --model=./inference/det_db_om/DBNet_MobileNetV3_bs${batchsize}.om \
          --loop=100 \
          --batchsize=${batchsize}

      ```

      -   参数说明：

          -   --model：om模型路径。
          -   --loop：推理次数。
          -   --batchsize：om模型的batch。

      `${batchsize}`表示不同batch的om模型。

      纯推理完成后，在ais_bench的屏显日志中`throughput`为计算的模型推理性能。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| ---------- | ----------- | ---------- | ----------- | ---------------- |
|Ascend310P3| 1           | ICDAR 2015 | Acc:0.775 | 196.102 fps |
|Ascend310P3| 4           | ICDAR 2015 | Acc:0.775 | 161.343 fps |
|Ascend310P3| 8           | ICDAR 2015 | Acc:0.775 | 158.040 fps |
|Ascend310P3| 16          | ICDAR 2015 | Acc:0.775 | 152.316 fps |
|Ascend310P3| 32          | ICDAR 2015 | Acc:0.775 | 152.485 fps |
|Ascend310P3| 64          | ICDAR 2015 | Acc:0.775 | 153.007 fps |
