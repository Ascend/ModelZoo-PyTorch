# ch_PP-OCRv2_rec模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

ch_PP-OCRv2_rec是基于[PP-OCRv2](https://arxiv.org/abs/2109.03144)的中文文本识别模型，PP-OCRv2在PP-OCR的基础上，进一步在5个方面重点优化，检测模型采用CML协同互学习知识蒸馏策略和CopyPaste数据增广策略；识别模型采用LCNet轻量级骨干网络、UDML 改进知识蒸馏策略和Enhanced CTC loss损失函数改进，进一步在推理速度和预测效果上取得明显提升。

- 参考实现：

  ```
  url=https://github.com/PaddlePaddle/PaddleOCR.git
  branch=release/2.5
  commit_id=a40f64a70b8d290b74557a41d869c0f9ce4959d5
  model_name=ch_PP-OCRv2_rec
  ```

  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x H x W | NCHW         |

- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | batchsize x D x 6625 | FLOAT32  | ND           |


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

1. 获取源码。

   ```
   git clone -b release/2.5 https://github.com/PaddlePaddle/PaddleOCR.git
   cd PaddleOCR 
   git reset --hard a40f64a70b8d290b74557a41d869c0f9ce4959d5
   git apply ../ch_PP-OCRv2_rec.patch
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

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用PaddleOCR提供的中文识别[样例图片](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.5/doc/imgs_words/ch)作为测试集，该样本图片在`ch_PP-OCRv2_rec/PaddleOCR/doc/imgs_words/ch/`目录下，包括5张图片样本，在线推理方式参考[文字识别模型评估与预测](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/recognition.md#32-%E6%B5%8B%E8%AF%95%E8%AF%86%E5%88%AB%E6%95%88%E6%9E%9C)。

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   在`ch_PP-OCRv2_rec`工作目录下，执行ch_PP-OCRv2_rec_preprocess.py脚本，完成预处理。

   ```
    python3 ch_PP-OCRv2_rec_preprocess.py \
        -c PaddleOCR/configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec_distillation.yml \
        -o Global.infer_img=PaddleOCR/doc/imgs_words/ch/
   ```

   - 参数说明：

       -   -c：模型配置文件。
       -   -o：可选参数列表: Global.infer_img表示图片路径。

   运行后在当前目录下的`pre_data`路径中保存生成的二进制数据。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用`paddle2onnx`将模型权重文件转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       训练模型链接为：https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar。

       推理模型链接为：https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar。

       在`ch_PP-OCRv2_rec`工作目录下可通过以下命令获取Paddle训练模型和推理模型。

       ```
       wget -nc -P ./checkpoint https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar
       cd ./checkpoint && tar xf ch_PP-OCRv2_rec_train.tar && cd ..

       wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar
       cd ./inference && tar xf ch_PP-OCRv2_rec_infer.tar && cd ..
       ```

   2. 导出onnx文件。

      1. 使用paddle2onnx工具导出onnx文件。

         在`ch_PP-OCRv2_rec`工作目录下通过运行以下命令获取onnx模型。

         ```
         paddle2onnx \
             --model_dir ./inference/ch_PP-OCRv2_rec_infer \
             --model_filename inference.pdmodel \
             --params_filename inference.pdiparams \
             --save_file ./ch_PP-OCRv2_rec.onnx \
             --opset_version 11 \
             --enable_onnx_checker True \
             --input_shape_dict="{'x':[-1,3,-1,-1]}"
         ```

         参数说明请通过`paddle2onnx -h`命令查看。
         运行后在`ch_PP-OCRv2_rec`目录下获得ch_PP-OCRv2_rec.onnx文件。

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
         ```
         atc --framework=5 \
             --model=./ch_PP-OCRv2_rec.onnx \
             --output=./ch_PP-OCRv2_rec_bs${batchsize} \
             --input_format=NCHW \
             --input_shape="x:${batchsize},3,-1,-1" \
             --dynamic_image_size="32,320;32,413" \
             --log=error  \
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
           -   --dynamic_image_size：设置输入图片的动态分辨率参数。适用于执行推理时，每次处理图片宽和高不固定的场景。

           `${batchsize}`表示om模型可支持不同batch推理，可取值为：1，4，8，16，32，64。

           运行成功后生成`ch_PP-OCRv2_rec_bs${batchsize}.om`模型文件。

2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  


   b.  执行推理。

      ```
      python3 ch_PP-OCRv2_rec_ais_infer.py \
          --ais_infer=${path_to_ais-infer}/ais_infer.py \
          --model=./ch_PP-OCRv2_rec_bs${batchsize}.om \
          --inputs=./pre_data \
          --batchsize=${batchsize}
      ```

      -   参数说明：
           -   --ais_infer：ais_infer.py脚本路径
           -   --model：om模型路径。
           -   --inputs：输入数据集路径。
           -   --batchsize：om模型输入的batchsize。

      `${path_to_ais-infer}`为ais_infer.py脚本所在路径。`${batchsize}`表示不同batch的om模型。。

      推理完成后结果保存在`ch_PP-OCRv2_rec/results_bs${batchsize}`目录下。


   c.  精度验证。

      执行后处理脚本`ch_PP-OCRv2_rec_postprocess.py`，参考命令如下：

      ```
      python3 ch_PP-OCRv2_rec_postprocess.py \
          -c PaddleOCR/configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec_distillation.yml \
          -o Global.infer_results=${results_path}
      ```

      -   参数说明：

            -   -c：模型配置文件。
            -   -o：可选参数：Global.infer_results表示om推理结果路径。

      ${results_path}为推理结果的保存路径。

      后处理结果通过终端显示，参考如下(以实际运行为准)：

      ```
      Infer Results:
      word_1.jpg :  {'Student': [('韩国小馆', 0.99609375)]}
      word_2.jpg :  {'Student': [('汉阳鹦鹉家居建材市场E区25-26号', 0.9956597089767456)]}
      word_3.jpg :  {'Student': [('电话：15952301928', 0.9916294813156128)]}
      word_4.jpg :  {'Student': [('实力活力', 0.97998046875)]}
      word_5.jpg :  {'Student': [('西湾监管', 0.92138671875)]}
      ```

      在线推理命令如下：
      
      ```
      python3 PaddleOCR/tools/infer_rec.py \
          -c PaddleOCR/configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec_distillation.yml \
          -o Global.pretrained_model=./checkpoint/ch_PP-OCRv2_rec_train/best_accuracy Global.infer_img=PaddleOCR/doc/imgs_words/ch/
      ```
      
      在线推理结果参考如下（以实际运行为准）：
      
      ```
      ppocr INFO: infer_img: PaddleOCR/doc/imgs_words/ch/word_1.jpg
      ppocr INFO:        result: {"Student": {"label": "韩国小馆", "score": 0.9977312684059143}, "Teacher": {"label": "韩国小馆", "score": 0.999502420425415}}
      ppocr INFO: infer_img: PaddleOCR/doc/imgs_words/ch/word_2.jpg
      ppocr INFO:        result: {"Student": {"label": "汉阳鹦鹉家居建材市场E区25-26号", "score": 0.9959088563919067}, "Teacher": {"label": "汉阳鹦鹉家居建材市场E区25-26号", "score": 0.9943523406982422}}
      ppocr INFO: infer_img: PaddleOCR/doc/imgs_words/ch/word_3.jpg
      ppocr INFO:        result: {"Student": {"label": "电话：15952301928", "score": 0.9918777346611023}, "Teacher": {"label": "电话：15952301928", "score": 0.9894357919692993}}
      ppocr INFO: infer_img: PaddleOCR/doc/imgs_words/ch/word_4.jpg
      ppocr INFO:        result: {"Student": {"label": "实力活力", "score": 0.9812984466552734}, "Teacher": {"label": "实力活力", "score": 0.9794982075691223}}
      ppocr INFO: infer_img: PaddleOCR/doc/imgs_words/ch/word_5.jpg
      ppocr INFO:        result: {"Student": {"label": "西湾监管", "score": 0.9254761338233948}, "Teacher": {"label": "西湾监管", "score": 0.9788504838943481}}
      ```


   d.  性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python -m ais_bench \
          --model=./ch_PP-OCRv2_rec_bss${batchsize}.om \
          --dymHW=32,320 \
          --loop=20 \
          --batchsize=${batchsize}
      ```

      -   参数说明：

          -   --model：om模型路径。
          -   --loop：推理次数。
          -   --dymHW：指定动态shape模型的H和W。
          -   --batchsize：om模型的batch。

      `${batchsize}`表示不同batch的om模型。

      纯推理完成后，在ais_bench的屏显日志中`throughput`为计算的模型推理性能，如下所示：

      ```
      [INFO] throughput 1000*batchsize(16)/NPU_compute_time.mean(5.840699934959412): 2739.3977054414777
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号   | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ------------ | ---------- | ---------- | --------------- |
|Ascend310P3| 1            | 样例图片 | 与在线推理结果一致 | 272.386 fps  |
|Ascend310P3| 4            | 样例图片 | 与在线推理结果一致 | 964.959 fps  |
|Ascend310P3| 8            | 样例图片 | 与在线推理结果一致 | 1700.879 fps |
|Ascend310P3| 16           | 样例图片 | 与在线推理结果一致 | 2739.397 fps |
|Ascend310P3| 32           | 样例图片 | 与在线推理结果一致 | 2561.762 fps |
|Ascend310P3| 64           | 样例图片 | 与在线推理结果一致 | 2458.243 fps |