# ch_PP-OCRv3_det模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)
- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)
- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)
- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

ch_PP-OCRv3_det是基于PP-OCRv3的中文文本检测模型，PP-OCRv3在PP-OCR2的基础上，进行共计9个方面的升级，打造出一款全新的、效果更优的超轻量OCR系统，全新升级的PP-OCRv3的整体的框架图检测模块仍基于DB算法优化，而识别模块不再采用CRNN，更新为IJCAI 2022最新收录的文本识别算法SVTR，进一步在推理速度和预测效果上取得明显提升。

- 参考实现：

  ```
  url=https://github.com/PaddlePaddle/PaddleOCR.git
  branch=release/2.6
  commit_id=274c216c6771a94807a34fb94377a1d7d674a69f
  model_name=ch_PP-OCRv3_det
  ```

  


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x imgH x imgW | NCHW         |

- 输出数据

  | 输出数据 | 数据类型     | 大小 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32 | batchsize x 1 x imgH x imgW  | NCHW           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| paddlepaddle                                                 | 2.3.1   | 仅支持x86服务器安装                                                           |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git
   cd PaddleOCR 
   git reset --hard 274c216c6771a94807a34fb94377a1d7d674a69f
   patch -p1 < ../ch_PP-OCRv3_det.patch
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   cd PaddleOCR
   pip3 install -r requirements.txt
   python3 setup.py install
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   精度测试数据集使用PaddleOCR提供的中文测试[样例集](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/doc/imgs)，该样例集的目录为`ch_PP-OCRv3_det/PaddleOCR/doc/imgs/`，包括20张图片样本，由于样本model_prod_flow_ch.png图片过大，可能会超出了算子计算范围，故将其移除。在`ch_PP-OCRv3_det`工作目录下执行如下命令获取样例集：

   ```
    cp -r ./PaddleOCR/doc/imgs/ ./
    cp -r ./PaddleOCR/ppocr/ ./
    cp -r ./PaddleOCR/tools/ ./
    rm -rf ./imgs/model_prod_flow_ch.png
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   在`ch_PP-OCRv3_det`工作目录下，执行ch_PP-OCRv3_det_preprocess.py脚本，完成预处理。

   ```
    python3 ch_PP-OCRv3_det_preprocess.py \
        -c PaddleOCR/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
        -o Global.infer_img=./imgs/ prep_dir=prep_data_dir
   ```

   - 参数说明：

       -   -c：模型配置文件。
       -   -o：可选参数列表: Global.infer_img表示图片路径。prep_dir表示预处理后的结果保存路径。

   运行后在当前目录下的`prep_data_dir`路径中保存生成的numpy数据。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用`paddle2onnx`将模型权重文件转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       训练模型（在线推理使用）链接为：https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar。
       
       推理模型链接为：https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar。

       在`ch_PP-OCRv3_det`工作目录下可通过以下命令获取训练模型、推理模型和辅助模型。

       ```
       wget -nc -P ./checkpoint https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar
       cd ./checkpoint && tar xf ch_PP-OCRv3_det_distill_train.tar && cd ..
       
       wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
       cd ./inference && tar xf ch_PP-OCRv3_det_infer.tar && cd ..
       ```

   2. 导出onnx文件。

      1. 使用paddle2onnx工具导出onnx文件。

         在`ch_PP-OCRv3_det`工作目录下通过运行以下命令获取onnx模型。

         ```
         paddle2onnx \
             --model_dir ./inference/ch_PP-OCRv3_det_infer \
             --model_filename inference.pdmodel \
             --params_filename inference.pdiparams \
             --save_file ./ch_PP-OCRv3_det.onnx \
             --opset_version 11 \
             --enable_onnx_checker True \
             --input_shape_dict="{'x':[-1,3,-1,-1]}"
         ```

         参数说明请通过`paddle2onnx -h`命令查看。
         运行后在`ch_PP-OCRv3_det`目录下获得ch_PP-OCRv3_det.onnx文件。

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
             --model=./ch_PP-OCRv3_det.onnx \
             --output=./ch_PP-OCRv3_det_bs1 \
             --input_format=ND \
             --input_shape="x:1,3,-1,-1" \
             --soc_version=Ascend${chip_name} \
             --dynamic_dims="736,736;736,800;736,960;736,992;736,1184;736,1248;736,1280;768,928;832,1536;992,736;1088,736;1184,736"
         ```
    
         - 参数说明：
    
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --dynamic_dims：设置输入图片的动态分辨率参数。适用于执行推理时，每次处理图片宽和高不固定的场景。
    
           运行成功后生成`ch_PP-OCRv3_det_bs1.om`模型文件。

2. 开始推理验证。

   a. 安装ais_bench推理工具。     

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。


   b. 执行推理。
      在当前目录下运行以下指令
      ```
      python -m ais_bench --model=ch_PP-OCRv3_det_bs1.om --input=./prep_data_dir/img_npy --output=./ --output_dirname=results_bs1 --auto_set_dymdims_mode=1 --outfmt=NPY
      ```

      -   参数说明：
           -   --model：om模型路径。
           -   --inputs：输入数据集路径。
           -   --batchsize：om模型输入的batchsize。
           -   --auto_set_dymdims_mode：设置自动匹配动态shape
           -   --outfmt：输出数据格式
      推理结果保存在当前目录的results_bs1文件夹下



   c. 精度验证。

      执行后处理脚本`ch_PP-OCRv3_det_postprocess.py`，参考命令如下：
    
      ```
      python3 ch_PP-OCRv3_det_postprocess.py \
          -c PaddleOCR/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
          -o Global.infer_img=imgs/ 
             res_dir=results_bs1 \
             save_dir=${output_path}/ \
             info_path=prep_data_dir/img_info.pkl
      ```
    
      -   参数说明：
    
            -   -c：模型配置文件。
            -   -o：可选参数：Global.infer_img表示样本图片路径，res_dir表示推理结果保存路径，save_dir表示后处理结果保存路径，
                            info_path表示预处理后的结果pkl文件的路径。
    
      `${output_path}`为精度验证结果的保存路径，`results_bs1`为推理结果的保存路径，命令执行完成后，每个推理结果对应的检测图片保存在  `checkpoints/det_db/det_results_Student/`目录下：
    
      在线推理命令如下：
    
      ```
      python3 PaddleOCR/tools/infer_det.py \
          -c PaddleOCR/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
          -o Global.infer_img="./imgs/" \
          Global.pretrained_model="./checkpoint/ch_PP-OCRv3_det_distill_train/best_accuracy"
      ```
      -   参数说明：
    
            -   -c：模型配置文件。
            -   -o：可选参数：Global.infer_img表示样本图片路径，Global.pretrained_model表示预训练模型路径。
            
      执行完成后在线推理结果保存在`ch_PP-OCRv3_det/checkpoints/det_db/det_results_Student/`目录下。
    
      可以将om后处理得到的样例图片的推理结果，与在线推理得到的样例图片的推理结果进行对比，观察文本检测框的效果，来验证om的推理精度。




# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------- | ---------- | ---------- | --------------- |
|Ascend310P3| 1          | 样例图片 | 见备注 | 215 fps |


   - 备注：将OM推理结果后处理后，与在线推理结果进行对比，对于每张验证图片，两者得到的文本框数量与位置均一致，可判定OM精度正常。
