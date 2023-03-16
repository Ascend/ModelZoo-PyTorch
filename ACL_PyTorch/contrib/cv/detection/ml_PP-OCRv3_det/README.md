# ml_PP-OCRv3_det模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)
- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)
- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)
- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

ml_PP-OCRv3_det是基于PP-OCRv3的混合语言文本检测模型，PP-OCRv3在PP-OCR2的基础上，进行共计9个方面的升级，打造出一款全新的、效果更优的超轻量OCR系统，全新升级的PP-OCRv3的整体的框架图检测模块仍基于DB算法优化，而识别模块不再采用CRNN，更新为IJCAI 2022最新收录的文本识别算法SVTR，进一步在推理速度和预测效果上取得明显提升。

- 参考实现：

  ```
  url=https://github.com/PaddlePaddle/PaddleOCR.git
  branch=release/2.6
  commit_id=274c216c6771a94807a34fb94377a1d7d674a69f
  model_name=ml_PP-OCRv3_det
  ```

  


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型  | 大小                         | 数据排布格式 |
  | -------- | -------- | --------------------------- | ------------ |
  | x        | RGB_FP32 | batchsize x 3 x imgH x imgW | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | sigmoid_0.tmp_0 | FLOAT32 | batchsize x 1 x -1 x -1 | NCHW           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.18  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 拉取源码并进行修改。

   ```bash
   git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git
   cd PaddleOCR 
   git reset --hard 274c216c6771a94807a34fb94377a1d7d674a69f
   patch -p1 < ../mlppocr.patch
   cd ..
   ```

2. 安装依赖。

   ```bash
   pip install -r requirements.txt
   pip install PaddleOCR
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   模型精度的验证使用PaddleOCR提供的测试图片，位于`PaddleOCR/doc/imgs/`目录下，包括20张图片样本，将其复制到当前目录并删除里面的非测试图片。
   ```bash
   cp -r ./PaddleOCR/doc/imgs/ ./
   rm -f ./imgs/model_prod_flow_ch.png
   ```

2. 数据预处理。

   执行数据预处理脚本，将原始图片转换成推理工具可以支持的npy文件。

   ```bash
   python mlppocr_preprocess.py \
       -c PaddleOCR/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
       -o Global.infer_img=./imgs/ prep_dir=prep_data
   ```

   - 参数说明：

       -   -c：模型配置文件。
       -   -o：可选参数列表: Global.infer_img表示图片路径。prep_data表示预处理后的结果保存路径。
   
   执行完以上命令后，每张原始图片都会对应生成一个npy文件存放于`prep_data/img_npy`目录下，此外还会生成一个`prep_data/img_info.pkl`文件，存放了一些每个图片的其他信息，用于后处理时绘制检测框。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用`paddle2onnx`将模型权重文件转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       下载该模型的[Paddle推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar)并解压，可参考命令：

       ```bash
       wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar
       tar xf Multilingual_PP-OCRv3_det_infer.tar
       ```

   2. 导出onnx文件。

       使用PaddleOCR内置的paddle2onnx工具，执行以下的命令导出onnx文件：

       ```bash
       paddle2onnx \
           --model_dir ./Multilingual_PP-OCRv3_det_infer \
           --model_filename inference.pdmodel \
           --params_filename inference.pdiparams \
           --save_file ./ml_PP-OCRv3_det.onnx \
           --opset_version 11 \
           --enable_onnx_checker True \
           --input_shape_dict="{'x':[-1,3,-1,-1]}"
       ```

       运行后在当前目录下会生成ml_PP-OCRv3_det.onnx文件。若想了解`paddle2onnx`工具的具体用法，可通过执行`paddle2onnx -h`命令或进入[paddle2onnx官方文档](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/deploy/paddle2onnx/readme_ch.md)查看。
       

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```bash
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 查看芯片名称（$\{chip\_name\}）。

         ```bash
         npu-smi info
         ```

         执行以上命令后，会打印以下形式的日志
         ```
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
         
         通过日志可中的Name关键字可查询到NPU芯片名为310P3，即：
         ```bash
         chip_name=310P3
         ```


      3. 执行ATC命令。
         ```bash
         bs=1  # 根据实际需要自行设置batchsize
         atc --framework=5 \
             --model=./ml_PP-OCRv3_det.onnx \
             --output=./ml_PP-OCRv3_det_bs${bs} \
             --input_format=NCHW \
             --input_shape="x:${bs},3,-1,-1" \
             --dynamic_image_size="736,736;736,800;736,960;736,992;736,1184;736,1248;736,1280;768,928;832,1536;992,736;1088,736;1184,736" \
             --log=error  \
             --soc_version=Ascend${chip_name} \
             --insert_op_conf=./mlppocr_aipp.cfg \
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
           -   --dynamic_image_size：设置输入图片的动态分辨率参数。适用于执行推理时，每次处理图片宽和高不固定的场景。
    
         可对bs设置不同的值以生成不同batchsize的OM模型，如上例bs=1，则会在当前目录下生成`ml_PP-OCRv3_det_bs1.om`。

2. 开始推理验证。

   OM模型使用ais_bench作为推理工具，其介绍、安装、使用、参数说明请参考ais_bench工具的[Gitee主页](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)。因为模型的特殊性，这里调用ais_benchd的推理接口进行推理。

   1. 执行推理。

      ```bash
      python mlppocr_inference.py \
          --model ml_PP-OCRv3_det_bs${bs}.om \
          --input prep_data/img_npy/ \
          --output om_results/bs${bs}
      ```
      - 参数说明：
        -   --model：OM模型文件的路径。
        -   --input：预处理过后的npy文件存放目录路径
        -   --output 推理结果的存放目录路径

      推理完成后，`./prep_data/img_npy`目录下的每个npy文件，在`om_results/bs${bs}`目录下都会对应生成一个npy文件，用于接收推理结果数据。

   2. 精度验证。
      
      此PaddleOCR模型，官方未提供可以用于对比的精度数据。为了验证转换后OM模型的推理精度，这里直接用肉眼比对OM离线推理结果与PaddleOCR模型在线推理结果，若一致，则可认为OM模型精度达标，否则精度不达标。

      首先执行后处理脚本，解析OM模型的离线推理结果，将检测框绘制到原图上。
    
      ```bash
      python mlppocr_postprocess.py \
          -c PaddleOCR/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
          -o Global.infer_img=./imgs/ res_dir=om_results/bs${bs} save_dir=om_results/bs${bs}_boxes \
             info_path=prep_data/img_info.pkl
      ```
    
      -   参数说明：
    
          -   -c：模型配置文件。
          -   -o：可选参数：Global.infer_img表示原始图片存放目录，res_dir表示推理结果存放目录，save_dir表示后处理结果存放目录，
                            info_path表示预处理后的结果pkl文件的路径。
    
      执行完以上命令后，`om_results/bs${bs}_boxes`目录下会生成绘制好检测框的图片。
    
      接下来下载PaddleOCR训练模型并解压，并对同一批测试图片进行在线推理：
    
      ```bash
      wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_distill_train.tar
      tar xf Multilingual_PP-OCRv3_det_distill_train.tar
      python PaddleOCR/tools/infer_det.py \
          -c PaddleOCR/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
          -o Global.infer_img=./imgs \
             Global.pretrained_model="./Multilingual_PP-OCRv3_det_distill_train/best_accuracy"
      ```

      -   参数说明：
    
          -   -c：模型配置文件。
          -   -o：可选参数：Global.infer_img表示样本图片路径，Global.pretrained_model表示预训练模型路径。
            
      执行完成后在线推理结果保存在`checkpoints/det_db/det_results_Student/`目录下。
    
      经人工比对`om_results/bs${bs}_boxes`与`checkpoints/det_db/det_results_Student/`两个目录，检测框数量和检测框位置均一致，以此判定OM模型**精度正常**。

   3. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
    
      ```bash
      python -m ais_bench --model=./ml_PP-OCRv3_det_bs${bs}.om --dymHW=736,992 --loop=100
      ```
    
      -   参数说明：
    
          -   --model：om模型路径。
          -   --loop：推理次数。
          -   --dymHW：动态分辨率参数，指定模型输入的实际H、W。
    
      纯推理完成后，找到日志中`throughput`关键字对应的数据，即为OM模型离线推理的吞吐率。需注意的是，此模型为接收的输入图片，宽与高是动态的，这里选用测试图片中出现频次最高的shape=(736, 992)来测试性能，选用其他shape会得到不同的性能指标。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

- 在310P3设备上对OM模型进行实验，各batch_size推理结果与PaddleOCR在线推理推理结果一致，由此判断精度达标；当batchsize为1时，模型性能最佳，达349.6fps。

    | 芯片型号   | Batch Size | 数据集  | 精度          | 性能        |
    | --------- | ---------- | ------- | ------------- | ---------- |
    |Ascend310P3| 1          | 样例图片 | 与在线推理一致 | 349.6 fps  |
    |Ascend310P3| 4          | 样例图片 | 与在线推理一致 | 339.1 fps  |
    |Ascend310P3| 8          | 样例图片 | 与在线推理一致 | 321.4 fps  |
    |Ascend310P3| 16         | 样例图片 | 与在线推理一致 | 323.2 fps  |
    |Ascend310P3| 32         | 样例图片 | 与在线推理一致 | 320.9 fps  |
    |Ascend310P3| 64         | 样例图片 | 与在线推理一致 | 286.8 fps  |
