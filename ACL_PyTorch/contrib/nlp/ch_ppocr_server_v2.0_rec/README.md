# ch_ppocr_server_v2.0_rec模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

  - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

ch_ppocr_server_v2.0_rec是一种通用的中文中文的识别模型，它的识别模块是基于文本识别算法SVTR优化。SVTR不再采用RNN结构，通过引入Transformers结构更加有效地挖掘文本行图像的上下文信息，从而提升文本识别能力。

- 参考实现：

  ```
  url=https://github.com/PaddlePaddle/PaddleOCR.git
  branch=release/2.6
  commit_id=7f6c9a7b99ea66077950238186137ec54f2b8cfd
  model_name=ch_ppocr_server_rec.v2.0
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                       | 数据排布格式 |
  | -------- |--------------------------| ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 32 x 320 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型     | 大小 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32 | batchsize x 40 x 97  | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | paddlepaddle                                                 | 2.3.2   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码,修改源码。

   ```
   git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git
   cd PaddleOCR 
   git reset --hard 7f6c9a7b99ea66077950238186137ec54f2b8cfd
   cd ..
   patch -p2 < ch_server_rec.v2.patch
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用PaddleOCR提供的中文识别[样例图片](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/doc/imgs_words/ch)作为测试集，该样本图片在`PaddleOCR/doc/imgs_words/ch/`目录下，包括5张图片样本，在线推理方式参考[文字识别模型评估与预测](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/recognition.md#32-%E6%B5%8B%E8%AF%95%E8%AF%86%E5%88%AB%E6%95%88%E6%9E%9C)，测试结果如下：
   ```
   result: 韩国小馆       0.9983214735984802
   result: 汉阳鹦鹉家居建材市场E区25-26号 0.9943711757659912
   result: 电话：15952301928      0.9942359924316406
   result: 实力活力       0.9992750287055969
   result: 西湾监管       0.9969249367713928
   ```

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   在`ch_ppocr_server_v2.0_rec`工作目录下，执行ch_server_rec_preprocess.py脚本，完成预处理。

   ```
    python3 ch_server_rec_preprocess.py \
        -c PaddleOCR/configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml \
        -o Global.infer_img=PaddleOCR/doc/imgs_words/ch/

   ```

   - 参数说明：

       -   -c：模型配置文件。
       -   -o：可选参数列表: Global.infer_img表示图片路径。

   运行后在当前目录下的`pre_data`路径中保存生成的二进制数据。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用export_model.py将训练模型转换为推理模型，再使用`paddle2onnx`将模型权重文件转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
       ```
       wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar
       tar xvf ch_ppocr_server_v2.0_rec_train.tar
       ```
       通过以下命令将获取的训练权重转为推理模型。
       ```
       python3 /PaddleOCR/tools/export_model.py 
               -c PaddleOCR/configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_train_v2.0.yml \
               -o Global.pretrained_model=ch_ppocr_server_v2.0_rec_train/best_accuracy \
               Global.save_inference_dir=ch_ppocr_server_v2.0_rec_infer/
       ```

   2. 导出onnx文件。

      1. 使用paddle2onnx工具导出onnx文件。

         在`ch_ppocr_server_v2.0_rec`工作目录下通过运行以下命令获取onnx模型。

         ```
         paddle2onnx \
             --model_dir ch_server_v2.0_rec_infer \
             --model_filename inference.pdmodel \
             --params_filename inference.pdiparams \
             --save_file ch_ppocr_server_v2.0_rec.onnx \
             --opset_version 12 \
             --input_shape_dict="{'x':[-1,3,32,-1]}" \
             --enable_onnx_checker True
         ```

         参数说明请通过`paddle2onnx -h`命令查看。
         运行后在当前目录下获得`ch_ppocr_server_v2.0_rec.onnx`文件。
      2. 执行以下命令修改onnx模型的domin
         ```
         python3 del_domin.py ./ch_ppocr_server_v2.0_rec.onnx ./ch_ppocr_server_v2.0_rec_new.onnx
         ```
         运行后在`onnx`目录下获得`ch_ppocr_server_v2.0_rec_new.onnx`文件

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
         在`ch_ppocr_server_v2.0_rec`目录下运行以下指令将onnx模型转换为om模型。
         ```
         atc --framework=5 \
             --model=./ch_ppocr_server_v2.0_rec_new.onnx \
             --output=./ch_ppocr_server_v2.0_rec_bs1 \
             --input_shape="x:1,3,-1,-1" \
             --log=error \
             --soc_version=${chip_name} \
             --dynamic_image_size="32,320;32,413"
         ```

         - 参数说明：

           - --model：为ONNX模型文件。
           - --framework：5代表ONNX模型。
           - --output：输出的OM模型。
           - --input\_format：输入数据的格式。
           - --input\_shape：输入数据的shape。
           - --log：日志级别。
           - --soc\_version：处理器型号。
           - --dynamic_image_size:设置输入图片的动态分辨率参数

           

2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请点击本链接进行安装ais_bench推理工具，以及查看具体使用方法(https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)  


   b.  执行推理。
      在当前目录下运行以下指令
      ```
      python3 ch_server_rec_ais_infer.py \
          --ais_infer=${path_to_ais-infer}/ais_infer.py \
          --model=./ch_ppocr_server_rec_bs${batchsize}.om \
          --inputs=./pre_data \
          --batchsize=${batchsize}
      ```

      -   参数说明：

           -   --model：om模型路径。
           -   --input：npy文件路径。


      推理完成后在当前`ch_ppocr_server_v2.0_rec`工作目录生成推理结果。其目录命名格式为`xxxx_xx_xx-xx_xx_xx`(`年_月_日-时_分_秒`)，如`2022_08_18-06_55_19`。

      >**说明：** 
      >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见。

   c.  精度验证。
      在`ch_ppocr_server_v2.0_rec`工作目录下执行后处理脚本`ch_server_rec_postprocess.py`，参考命令如下：

      ```
      python3 ch_server_rec_postprocess.py \
          -c PaddleOCR/configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml \
          -o Global.infer_results=./
      ```

      -   参数说明：

            -   -c：模型配置文件。
            -   -o：可选参数：Global.infer_results表示om推理结果路径。

      推理结果通过终端显示，如下：

      ```
      Infer Results:  {'word_1.png': ('韩国小馆', 0.998046875), 'word_2.png': ('汉阳鹦鹉家居建材市场E区25-26号', 0.9932725429534912), 'word_3.png': ('电话：15952301928', 0.9931640625), 'word_4.png': ('实力活力', 0.998046875), 'word_5.png': ('西湾监管', 0.99609375)}
      ```

   d.  性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench \
          --model=./ch_ppocr_server_v2.0_rec_bs${bs}.om \
          --dymHW=32,320 \
          --loop=100 \
          --batchsize=${bs}
      ```

      -   参数说明：

          -   --model：om模型路径。
          -   --loop：推理次数。
          -   --batchsize：om模型的batch。

      纯推理完成后，在ais_bench的屏显日志中`throughput`为计算的模型推理性能，如下所示：

      ```
       throughput 1000*batchsize(16)/NPU_compute_time.mean(11.091549987792968): 1634.0495925374837
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号   | Batch Size   | 数据集 | 精度 | 性能            |
| --------- | ------------ | ---------- | ---------- |---------------|
|Ascend310P3| 1            | 样例图片 | 与在线推理结果一致 | 294.77 fps    |
|Ascend310P3| 4            | 样例图片 | 与在线推理结果一致 | 896.68 fps   |
|Ascend310P3| 8            | 样例图片 | 与在线推理结果一致 | 1244.42 fps  |
|Ascend310P3| 16           | 样例图片 | 与在线推理结果一致 | 1634.04 fps  |
|Ascend310P3| 32           | 样例图片 | 与在线推理结果一致 | 591.15 fps |
|Ascend310P3| 64           | 样例图片 | 与在线推理结果一致 | 592.75 fps |
