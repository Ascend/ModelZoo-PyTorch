# ch_ppocr_mobile_v2.0_cls模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  ******




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

ch_ppocr_mobile_v2.0_cls为[[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/angle_class.md#%E6%96%B9%E6%B3%95%E4%BB%8B%E7%BB%8D)]内置的中文文本方向分类器，支持了0和180度的分类。

- 参考实现：

  ```
  url=https://github.com/PaddlePaddle/PaddleOCR.git
  branch=release/2.5
  commit_id=a40f64a70b8d290b74557a41d869c0f9ce4959d5
  model_name=ch_ppocr_mobile_v2.0_cls
  ```

- 通过Git获取对应commit\_id的代码方法如下：

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
  | input    | RGB_FP32 | batchsize x 3 x 48 x 192 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 2 | FLOAT32  | ND           |

# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| paddlepaddle                                                 | 2.3.1   | -                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

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

该模型在PaddleOCR提供的中文文本方向分类[[样本集](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.5/doc/imgs_words/ch)]进行精度验证，该样本集在`ch_ppocr_mobile_v2.0_cls/PaddleOCR/doc/imgs_words/ch`目录下，包括5张图片样本，其[[在线推理](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/angle_class.md#6-%E9%A2%84%E6%B5%8B)]结果如下：

  **表 2**  图片样本在线推理结果

| 图片       | 结果             | 
| ---------- | ---------------- |
| word_1.jpg | ('0', 0.9998784) |
| word_2.jpg | ('0', 1.0)       |
| word_3.jpg | ('0', 1.0)       |
| word_4.jpg | ('0', 0.9999982) |
| word_5.jpg | ('0', 0.9999988) |

2. 数据预处理。

数据预处理将原始数据集转换为模型输入的数据。

在`ch_ppocr_mobile_v2.0_cls`工作目录下，执行en_PP-OCRv3_rec_preprocess.py脚本，完成预处理。

   ```
    python3 ch_ppocr_mobile_v2.0_cls_preprocess.py \
        -c PaddleOCR/configs/cls/cls_mv3.yml \
        -o Global.infer_img=PaddleOCR/doc/imgs_words/ch
   ```

   - 参数说明：

       -   -c：模型配置文件。
       -   -o：可选参数列表: Global.infer_img表示图片路径。

运行后在`ch_ppocr_mobile_v2.0_cls/pre_data`路径中保存生成的二进制数据。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用`paddle2onnx`将模型权重文件转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      推理权重链接为：https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar。

      在`ch_ppocr_mobile_v2.0_cls`工作目录下可通过以下命令获取推理权重。

      ```
       wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
       cd ./inference && tar xf ch_ppocr_mobile_v2.0_cls_infer.tar && cd ..
      ```

   2. 导出onnx文件。

      1. 使用paddle2onnx工具导出onnx文件。

         在`ch_ppocr_mobile_v2.0_cls`工作目录下通过运行以下命令获取onnx文件。

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

         参数说明请通过`paddle2onnx -h`命令查看。
         运行后在`ch_ppocr_mobile_v2.0_cls`目录下获得`ch_ppocr_mobile_v2.0_cls.onnx`文件。

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
             --model=./ch_ppocr_mobile_v2.0_cls.onnx \
             --output=./ch_ppocr_mobile_v2.0_cls_bs${batchsize} \
             --input_format=NCHW \
             --input_shape="x:${batchsize},3,48,192" \
             --log=error \
             --soc_version=Ascend${chip_name} \
             --insert_op_conf=./aipp_ch_ppocr_mobile_v2.0_cls.config \
             --enable_small_channel=1
         ```

         `${batchsize}`表示om模型可支持不同batch推理，可取值为：1，4，8，16，32，64。

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

          运行成功后生成`ch_ppocr_mobile_v2.0_cls_bs${batchsize}.om`模型文件。

2. 开始推理验证。

a.  使用ais-infer工具进行推理。

执行ais-infer工具推理命令，如下：

   ```
    python3 ${path_to_ais-infer}/ais_infer.py \
        --model=./ch_ppocr_mobile_v2.0_cls_bs${batchsize}.om \
        --input=./pre_data \
        --output=./
   ```

`${path_to_ais-infer}`为ais_infer.py脚本的存放路径。`${batchsize}`表示不同batch的om模型。
   -   参数说明：

       -   --model：om模型路径。
       -   --input：bin文件路径。
       -   --output：推理结果保存路径。

推理完成后在当前`ch_ppocr_mobile_v2.0_cls`工作目录生成推理结果。其目录命名格式为`xxxx_xx_xx-xx_xx_xx`(`年_月_日-时_分_秒`)，如`2022_12_04-15_46_57`。


b.  精度验证。

执行后处理脚本`ch_ppocr_mobile_v2.0_cls_postprocess.py`，参考命令如下：

   ```
    python3 ch_ppocr_mobile_v2.0_cls_postprocess.py --config=PaddleOCR/configs/cls/cls_mv3.yml --opt=results=${output_path}
   ```
${output_path}为推理结果的保存路径。

   -   参数说明：

       -   --config：模型配置文件。
       -   --opt：推理结果路径。

推理结果通过屏显显示，如下：

   ```
    {'word_1.jpg': [('0', 0.9980469)], 
     'word_2.jpg': [('0', 0.9980469)], 
     'word_3.jpg': [('0', 0.9980469)], 
     'word_4.jpg': [('0', 0.9980469)], 
     'word_5.jpg': [('0', 0.9980469)]}
   ```

c.  性能验证。

可以使用ais-infer工具的纯推理模式验证模型性能，命令如下。

   ```
    python3 ${path_to_ais-infer}/ais_infer.py \
        --model=./ch_ppocr_mobile_v2.0_cls_bs${batchsize}.om \
        --loop=50 \
        --batchsize=${batchsize}
   ```

 `${path_to_ais-infer}`为ais_infer.py脚本的存放路径。`${batchsize}`表示不同batch的om模型。

   -   参数说明：

       -   --model：om模型路径。
       -   --loop：推理次数。
       -   --batchsize：om模型的batch。
    
纯推理完成后，在ais-infer的屏显日志中`throughput`为计算的模型推理性能。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|Ascend310P3| 1                | 样例图片 | 与在线推理一致 | 1756.851  fps |
|Ascend310P3| 4                | 样例图片 | 与在线推理一致 | 7367.839  fps |
|Ascend310P3| 8                | 样例图片 | 与在线推理一致 | 13141.683 fps |
|Ascend310P3| 16               | 样例图片 | 与在线推理一致 | 18906.942 fps |
|Ascend310P3| 32               | 样例图片 | 与在线推理一致 | 24095.478 fps |
|Ascend310P3| 64               | 样例图片 | 与在线推理一致 | 31958.773 fps |