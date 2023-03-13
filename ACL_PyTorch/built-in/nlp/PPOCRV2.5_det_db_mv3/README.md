# det_mv3_db_v2.0 模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

分割在文本检测中很流行，而二值化的后处理对于分割的检测至关重要，该检测方法将由分割方法生成的概率图转换为文本的边界框/区域。在该网络中，我们提出了一个名为可微分二值化（DB）的模块，它可以在分割网络中执行二值化过程。与DB模块一起优化的分割网络可以自适应地设置用于二值化的阈值，这不仅简化了后处理，还提高了文本检测的性能。

- 参考实现：

  ```
  url=https://github.com/PaddlePaddle/PaddleOCR.git
  branch=release/2.5
  commit_id=7f2d05cfe4e
  model_name=PPOCRV2.5_det_db_mv3 
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据
 
  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | 24 x 3 x 736 x 1280 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型    | 大小 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32 | 24 x 1 x 736 x 1280  | NCHW           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1.2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| paddlepaddle                                                 | 2.3.1   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone -b release/2.5 https://github.com/PaddlePaddle/PaddleOCR.git
   cd PaddleOCR 
   git reset --hard 7f2d05cfe4e
   patch -p1 < ../db.patch
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   cd PaddleOCR
   python3 setup.py install
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用ICDAR2015数据集，其处理方式[参考](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/dataset/ocr_datasets.md),处理完后的目录结构如下:

   ```
   /PaddleOCR/train_data/icdar2015/text_localization/
      └─ icdar_c4_train_imgs/         icdar 2015 数据集的训练数据
      └─ ch4_test_images/             icdar 2015 数据集的测试数据
      └─ train_icdar2015_label.txt    icdar 2015 数据集的训练标注
      └─ test_icdar2015_label.txt     icdar 2015 数据集的测试标注
   ```



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用`paddle2onnx`将模型权重文件转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       训练模型链接为：https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar

       
       在`PaddleOCR`工作目录下可通过以下命令获取Paddle训练模型和推理模型。

       ```
       wget -nc -P ./checkpoint https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar
       cd ./checkpoint && tar xf det_mv3_db_v2.0_train.tar && cd ..

       ```
   2. 转换成推理权重

      ```
      python3 tools/export_model.py -c configs/det/det_mv3_db.yml -o Global.pretrained_model=./checkpoint/det_mv3_db_v2.0_train/best_accuracy  Global.save_inference_dir=./inference/db_mv3 Global.use_gpu=False
      ```

   3. 导出onnx文件。

      1. 使用paddle2onnx工具导出onnx文件。

         在`PaddleOCR`工作目录下通过运行以下命令获取onnx模型。

         ```
         paddle2onnx \
         --model_dir ./inference/db_mv3 \
         --model_filename inference.pdmodel \
         --params_filename inference.pdiparams \
         --save_file ./db_mv3.onnx \
         --opset_version 11 \
         --enable_onnx_checker True \
         --input_shape_dict="{'x':[-1,3,-1,-1]}"
         ```

         参数说明请通过`paddle2onnx -h`命令查看。

         运行后在`PaddleOCR`目录下获得`db_mv3.onnx`文件。

   4. 使用ATC工具将ONNX模型转OM模型。

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
      
         在`PaddleOCR`目录执行
         ```
         atc --framework=5 --model=db_mv3.onnx --output=om/db_mv3_24bs \
         --input_format=NCHW --input_shape="x:24,3,736,1280" \
         --log=error --op_select_implmode=high_performance --optypelist_for_implmode=Sigmoid \
         --soc_version=Ascend${chip_name} --enable_small_channel=1 --insert_op_conf=aipp_dbnet.cfg
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --op_select_implmode: 算子模式的选择
           -   --optypelist_for_implmode：具体哪个算子
           -   --enable_small_channel: 使能C0特效
           -   --insert_op_conf：aipp配置文件


         运行成功后在`om`文件夹内生成`db_mv3_24bs.om`模型文件。

2. 开始推理验证。

   a. 安装ais_bench推理工具。

      请访问[ais_bench推理工具代码仓](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)，根据readme文档进行工具安装。


   b.  执行推理并验证精度

      在`PaddleOCR`目录执行

      ```
      python3 eval_npu.py -c ./configs/det/det_mv3_db.yml -o Global.use_gpu=False Global.device_id=0 Global.om_path=om/db_mv3_24bs.om Global.save_npu_path=npu_result Global.batch_size=24
      ```

      -   参数说明：
           -   -c：配置文件
           -   Global.use_gpu：是否使能gpu
           -   Global.device_id：选择npu的device_id
           -   Global.om_path：om文件路径
           -   Global.save_npu_path: 推理结果保存路径
           -   Global.batch_size: 模型推理bs

      推理完成后结果打屏显示

      >**说明：** 
      >执行ais-infer工具请选择与运行环境架构相同的命令

   
   c. 纯推理验证性能方式如下

      ```
      python3 -m ais_bench --model {om_path} --loop {loop} --batchsize {batchsize}
      ```
      + 参数说明

         - --model: om模型路径
         - --loop: 循环次数
         - --batchsize: 模型batchsiz

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号   | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ------------ | ---------- | ---------- | --------------- |
|310P3| 24           | ICDAR2015 | 73% | 187  |