# SAST模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)





# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

SAST模型提出了一个one-shot的文本检测器，基于多任务学习，针对任意形状包括多方向、多语言、弯曲场景文本，并且在速度上足够快。上下文注意力模块Content-Attention-Block聚合信息，以增加特征表示，而且不需要额外的计算开销。点到四边对齐的方法在鲁棒性和准确性方面相比较连通域分析都具有一定的优势，能够减缓文本被分块的问题。




- 参考实现：

  ```
	url=https://github.com/PaddlePaddle/PaddleOCR.git
	branch=release/2.5
	commit_id=a40f64a70b8d290b74557a41d869c0f9ce4959d5
	model_name=SAST
  ```
  






## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 896 x 1536 | NCHW         |


- 输出数据

   | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
   | -------- | -------- | -------- | ------------ |
   | output1  | FLOAT32  | batchsize x 4 x 224 x 384 | NCHW           |
   | output2  | FLOAT32  | batchsize x 1 x 224 x 384 | NCHW           |
   | output3  | FLOAT32  | batchsize x 2 x 224 x 384 | NCHW           |
   | output4  | FLOAT32  | batchsize x 8 x 224 x 384 | NCHW           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.11.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |





# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/PaddlePaddle/PaddleOCR.git
   cd PaddleOCR  
   git checkout release/2.5
   git reset --hard  a40f64a70b8d290b74557a41d869c0f9ce4959d5    
   cd ..
   patch -p2 < sast.patch
   export PYTHONPATH=./PaddleOCR
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```
   >PaddlePaddle目前暂不支持arm64框架

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   ICDAR 2015 数据集包含1000张训练图像和500张测试图像。参考[[PaddleOCR数据集](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/dataset/ocr_datasets.md)]数据处理方式，ICDAR 2015 数据集可以点击[[链接](https://rrc.cvc.uab.es/?ch=4&com=downloads)]进行下载。

	将数据集`ch4_test_images.zip`放在`SAST`工作目录下，通过以下命令创建`train_data/icdar2015/text_localization`路径，将下载的数据集保存该路径下，并在该路径下通过以下命令进行解压保存并获取标签文件。
   ```
   mkdir -p ./train_data/icdar2015/text_localization/ch4_test_images/
   unzip -d ./train_data/icdar2015/text_localization/ch4_test_images/ ch4_test_images.zip
   wget -P ./train_data/icdar2015/text_localization/ https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。。

   执行sast_preprocess.py脚本，完成预处理。

   ```
    python3 sast_preprocess.py \
        --config=PaddleOCR/configs/det/det_r50_vd_sast_icdar15.yml \
        --opt=bin_data=./icda2015_bin
   ```
   - 参数说明：
       -   --config：模型配置文件。
       -   --opt：bin文件保存路径。

   运行后在当前目录下的`icda2015_bin`路径中保存生成的二进制数据。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       训练权重链接为：https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar。
       在`SAST`工作目录下可通过以下命令获取训练权重并转为推理模型。
       ```
       wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar
       tar -xvf det_r50_vd_sast_icdar15_v2.0_train.tar
       python3 PaddleOCR/tools/export_model.py \
				-c PaddleOCR/configs/det/det_r50_vd_sast_icdar15.yml \
				-o Global.pretrained_model=./det_r50_vd_sast_icdar15_v2.0_train/best_accuracy \
				Global.save_inference_dir=./sast
       ```
      
       - 参数说明：

            -   -c：模型配置文件。
            -   -o: 模型入参信息。
            -   Global.pretrained_model：权重文件保存路径。
            -   Global.save_inference_dir：paddleocr推理模型保存路径。

   2. 导出onnx文件。

      1. 使用paddle2onnx工具onnx文件。

         在`SAST`工作目录下通过运行以下命令获取onnx模型。

         ```
         paddle2onnx \
             --model_dir ./sast \
             --model_filename inference.pdmodel \
             --params_filename inference.pdiparams \
             --save_file ./sast.onnx \
             --opset_version 11 \
             --input_shape_dict="{'x':[-1,3,896,1536]}"
         ```

         参数说明请通过`paddle2onnx -h`命令查看。
         运行后在`SAST`目录下获得sast.onnx文件。


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
					--model=./sast.onnx \
					--output=./sast_bs${batchsize} \
					--input_format=NCHW \
					--input_shape="x:${batchsize},3,896,1536" \
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


           运行成功后生成<u>***sast_bs${batchsize}.om***</u>模型文件。

2. 开始推理验证。

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
		python3 ${path_to_ais-infer}/ais_infer.py \
				--model=./sast_bs${batchsize}.om \
				--input=./icda2015_bin \
				--output=./ 
            --batchsize=${batchsize} 
        ```

        -   参数说明：

             -   model：om文件路径。
             -   input：预处理数据地址。
             -   output：推理结果保存地址。


        >**说明：** 
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]。

   3. 精度验证。

      执行后处理脚本sast_postprocess.py`，参考命令如下：

      ```
      python3 sast_postprocess.py \
             --config=PaddleOCR/configs/det/det_r50_vd_sast_icdar15.yml \
             --opt=results=${time_line}
      ```

      -   参数说明：

            -   --config：模型配置文件。
            -   --opt：推理结果路径。


   4. 性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 ${ais_infer_path}/ais_infer.py --model=sast_bs${bs} --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型地址
        - --batchsize：数据的batchsize



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|    Ascend310P3       |      1            |      ICDAR 2015      |     91.3%       |      19.94           |
|    Ascend310P3       |      4            |      ICDAR 2015      |           |       25.26          |
|    Ascend310P3       |      8            |      ICDAR 2015      |         |          25.39       |
|    Ascend310P3       |      16            |      ICDAR 2015      |     91.3%       |    25.04             |
|    Ascend310P3       |      32            |      ICDAR 2015      |          |         25.22        |
|    Ascend310P3       |      64            |      ICDAR 2015      |    超出内存    |                 |
