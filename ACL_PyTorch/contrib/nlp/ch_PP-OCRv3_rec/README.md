# ch_PP-OCRv3_rec模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

ch_PP-OCRv3_rec是基于[[PP-OCRv3](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/PP-OCRv3_introduction.md)]的中文文本识别模型，PP-OCRv3的识别模块是基于文本识别算法SVTR优化。SVTR不再采用RNN结构，通过引入Transformers结构更加有效地挖掘文本行图像的上下文信息，从而提升文本识别能力。

- 参考实现：

  ```
  url=https://github.com/PaddlePaddle/PaddleOCR.git
  branch=release/2.5
  commit_id=a40f64a70b8d290b74557a41d869c0f9ce4959d5
  model_name=ch_PP-OCRv3_rec
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
  | input    | RGB_FP32 | batchsize x 3 x 48 x W | NCHW         |

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
   git apply ../ch_PP-OCRv3_rec.patch
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

   该模型使用PaddleOCR提供的中文识别[样例图片](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.5/doc/imgs_words/ch)作为测试集，该样本图片在`ch_PP-OCRv3_rec/PaddleOCR/doc/imgs_words/ch/`目录下，包括5张图片样本，在线推理方式参考[文字识别模型评估与预测](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/recognition.md#32-%E6%B5%8B%E8%AF%95%E8%AF%86%E5%88%AB%E6%95%88%E6%9E%9C)。

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   在`ch_PP-OCRv3_rec`工作目录下，执行ch_PP-OCRv3_rec_preprocess.py脚本，完成预处理。

   ```
    python3 ch_PP-OCRv3_rec_preprocess.py \
        -c PaddleOCR/configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml \
        -o Global.infer_img=PaddleOCR/doc/imgs_words/ch/ Global.bin_data=./image_bin

   ```

   - 参数说明：

       -   -c：模型配置文件。
       -   -o：可选参数列表: Global.infer_img表示图片路径，Global.bin_data表示bin文件保存路径。

   运行后在当前目录下的`image_bin`路径中保存生成的二进制数据。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用`paddle2onnx`将模型权重文件转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       训练模型链接为：https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar。

       推理模型链接为：https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar。

       在`ch_PP-OCRv3_rec`工作目录下可通过以下命令获取Paddle训练模型和推理模型。

       ```
       wget -nc -P ./checkpoint https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar
       cd ./checkpoint && tar xf ch_PP-OCRv3_rec_train.tar && cd ..

       wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
       cd ./inference && tar xf ch_PP-OCRv3_rec_infer.tar && cd ..
       ```

   2. 导出onnx文件。

      1. 使用paddle2onnx工具导出onnx文件。

         在`ch_PP-OCRv3_rec`工作目录下通过运行以下命令获取onnx模型。

         ```
         paddle2onnx \
             --model_dir ./inference/ch_PP-OCRv3_rec_infer \
             --model_filename inference.pdmodel \
             --params_filename inference.pdiparams \
             --save_file ./ch_PP-OCRv3_rec.onnx \
             --opset_version 11 \
             --enable_onnx_checker True \
             --input_shape_dict="{'x':[-1,3,-1,-1]}"
         ```

         参数说明请通过`paddle2onnx -h`命令查看。
         运行后在`ch_PP-OCRv3_rec`目录下获得ch_PP-OCRv3_rec.onnx文件。

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
             --model=./ch_PP-OCRv3_rec.onnx \
             --output=./ch_PP-OCRv3_rec_dybs_${imgW} \
             --input_format=NCHW \
             --input_shape="x:-1,3,48,${imgW}" \
             --dynamic_batch_size="1,4,8,16,32,64" \
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

           `${imgW}`表示om模型可支持不同图片宽度的推理，取值为：320，620。
           运行成功后生成`ch_PP-OCRv3_rec_dybs_${imgW}.om`模型文件。

2. 开始推理验证。

   a.  使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]


   b.  执行推理。

      推理脚本命令格式如下：
      ```
      bash ais_infer.sh ${ais_infer_path} ${imgW320_om} ${imgW620_om} ${bin_data} ${results_path} ${batchsize}
      ```

      -   参数说明：

           -   ${ais_infer_path}：ais_infer python脚本路径。
           -   ${imgW320_om}：imgW为320的om。
           -   ${imgW620_om}：imgW为620的om。
           -   ${bin_data}：bin数据路径。
           -   ${results_path}：推理结果保存路径。
           -   ${batchsize}：设置不同batchsize，可支持：1，4，8，16，32，64。
      
      推理脚本命令举例：

      ```
      bash ais_infer.sh ./tools/ais_infer/ais_infer.py ./ch_PP-OCRv3_rec_dybs_320.om ./ch_PP-OCRv3_rec_dybs_620.om ./image_bin ./result_bs1 1
      ```

      命令运行结束后在`./result_bs1`目录下生成batchsize=1时的推理结果，获取其他batchsize的推理结果只需要在命令中修改`${results_path}`和`${batchsize}`两个参数即可。

      >**说明：** 
      >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   c.  精度验证。

      执行后处理脚本`ch_PP-OCRv3_rec_postprocess.py`，参考命令如下：

      ```
      python3 ch_PP-OCRv3_rec_postprocess.py \
          -c PaddleOCR/configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml \
          -o Global.infer_results=${results_path}
      ```

      -   参数说明：

            -   -c：模型配置文件。
            -   -o：可选参数：Global.infer_results表示om推理结果路径。

      ${results_path}为推理结果的保存路径。

      后处理结果通过终端显示，参考如下(以实际运行为准)：

      ```
      Infer Results:
      word_1.jpg :  {'Student': [('韩国小馆', 0.994140625)]}
      word_2.jpg :  {'Student': [('汉阳鹦鹉家居建材市场E区25-26号', 0.9690755009651184)]}
      word_3.jpg :  {'Student': [('电话：15952301928', 0.9037388563156128)]}
      word_4.jpg :  {'Student': [('实力活力', 0.99609375)]}
      word_5.jpg :  {'Student': [('西湾监管', 0.9951171875)]}
      ```

      在线推理命令如下：
      
      ```
      python3 PaddleOCR/tools/infer_rec.py \
          -c PaddleOCR/configs/rec/PP-OCRv3/ch_PP-OCRv3_rec_distillation.yml \
          -o Global.pretrained_model=./checkpoint/ch_PP-OCRv3_rec_train/best_accuracy Global.infer_img=PaddleOCR/doc/imgs_words/ch/
      ```
      
      在线推理结果如下（以实际运行为准）：
      
      ```
      ppocr INFO: infer_img: PaddleOCR/doc/imgs_words/ch/word_1.jpg
      ppocr INFO:        result: {"Student": {"label": "韩国小馆", "score": 0.9944804310798645}, "Teacher": {"label": "韩国小馆", "score": 0.9819368124008179}}
      ppocr INFO: infer_img: PaddleOCR/doc/imgs_words/ch/word_2.jpg
      ppocr INFO:        result: {"Student": {"label": "汉阳鹦鹉家居建材市场E区25-26号", "score": 0.9687681198120117}, "Teacher": {"label": "汉阳鹦鹉家居建材市场E区25-26号", "score": 0.9344552159309387}}
      ppocr INFO: infer_img: PaddleOCR/doc/imgs_words/ch/word_3.jpg
      ppocr INFO:        result: {"Student": {"label": "电话：15952301928", "score": 0.9042935371398926}, "Teacher": {"label": "电话：15952301928", "score": 0.9218837022781372}}
      ppocr INFO: infer_img: PaddleOCR/doc/imgs_words/ch/word_4.jpg
      ppocr INFO:        result: {"Student": {"label": "实力活力", "score": 0.9956860542297363}, "Teacher": {"label": "实力活力", "score": 0.9920592308044434}}
      ppocr INFO: infer_img: PaddleOCR/doc/imgs_words/ch/word_5.jpg
      ppocr INFO:        result: {"Student": {"label": "西湾监管", "score": 0.9959428310394287}, "Teacher": {"label": "西湾监管", "score": 0.9947077035903931}}
      ```
      
      将后理的om推理结果与在线推理结果进行对比。

   d.  性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 ${path_to_ais-infer}/ais_infer.py \
          --model=./ch_PP-OCRv3_rec_dybs_320.om \
          --loop=50 \
          --dymBatch=${batchsize} \
          --batchsize=${batchsize}
      ```

      -   参数说明：

          -   --model：om模型路径。
          -   --loop：推理次数。
          -   --dymBatch：om模型的batch。
          -   --batchsize：om模型的batch。

      `${path_to_ais-infer}`为ais_infer.py脚本的存放路径。`${batchsize}`表示不同batch的om模型。

      纯推理完成后，在ais-infer的屏显日志中`throughput`为计算的模型推理性能，如下所示（仅供参考，以实现推理性能为准）：

      ```
       [INFO] throughput 1000*batchsize(16)/NPU_compute_time.mean(3.30181999206543): 4845.8123212196415
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号   | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ------------ | ---------- | ---------- | --------------- |
|Ascend310P3| 1            | 样例图片 | 与在线推理结果一致 | 1547.795 fps |
|Ascend310P3| 4            | 样例图片 | 与在线推理结果一致 | 3639.606 fps |
|Ascend310P3| 8            | 样例图片 | 与在线推理结果一致 | 4499.437 fps |
|Ascend310P3| 16           | 样例图片 | 与在线推理结果一致 | 4845.812 fps |
|Ascend310P3| 32           | 样例图片 | 与在线推理结果一致 | 4293.146 fps |
|Ascend310P3| 64           | 样例图片 | 与在线推理结果一致 | 4153.195 fps |