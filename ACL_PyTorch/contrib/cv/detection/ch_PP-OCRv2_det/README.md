# ch_PP-OCRv2_det模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

ch_PP-OCRv2_det是基于PP-OCRv2的中文文本检测模型，PP-OCRv2在PP-OCR的基础上，进一步在5个方面重点优化，检测模型采用CML协同互学习知识蒸馏策略和CopyPaste数据增广策略；识别模型采用LCNet轻量级骨干网络、UDML 改进知识蒸馏策略和Enhanced CTC loss损失函数改进，进一步在推理速度和预测效果上取得明显提升。

- 参考实现：

  ```
  url=https://github.com/PaddlePaddle/PaddleOCR.git
  branch=release/2.6
  commit_id=274c216c6771a94807a34fb94377a1d7d674a69f
  model_name=ch_PP-OCRv2_det
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
  | input    | RGB_FP32 | batchsize x 3 x imgH x imgW | NCHW         |

- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | batchsize x 1 x imgH x imgW | FLOAT32  | NCHW           |


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
   git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git
   cd PaddleOCR 
   git reset --hard 274c216c6771a94807a34fb94377a1d7d674a69f
   git apply ../ch_PP-OCRv2_det.patch
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

1. 获取原始数据集。

   精度测试数据集使用PaddleOCR提供的中文测试[样例集](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/doc/imgs)，该样例集的目录为`ch_PP-OCRv2_det/PaddleOCR/doc/imgs/`，包括20张图片样本，由于样本model_prod_flow_ch.png图片过大，可能会超出了算子计算范围，故将其移除。在`ch_PP-OCRv2_det`工作目录下执行如下命令获取样例集：

   ```
    cp -r ./PaddleOCR/doc/imgs/ ./
    rm -rf ./imgs/model_prod_flow_ch.png
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   在`ch_PP-OCRv2_det`工作目录下，执行ch_PP-OCRv2_det_preprocess.py脚本，完成预处理。

   ```
    python3 ch_PP-OCRv2_det_preprocess.py \
        -c PaddleOCR/configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_cml.yml \
        -o Global.infer_img=./imgs/
   ```

   - 参数说明：

       -   -c：模型配置文件。
       -   -o：可选参数列表: Global.infer_img表示图片路径。

   运行后在当前目录下的`pre_data`路径中保存生成的numpy数据。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用`paddle2onnx`将模型权重文件转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       训练模型（在线推理使用）链接为：https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar。
       
       推理模型链接为：https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar。

       辅助训练模型（在线推理使用）链接为：https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar

       在`ch_PP-OCRv2_det`工作目录下可通过以下命令获取训练模型、推理模型和辅助模型。

       ```
       wget -nc -P ./checkpoint https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar
       cd ./checkpoint && tar xf ch_PP-OCRv2_det_distill_train.tar && cd ..

       wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar
       cd ./inference && tar xf ch_PP-OCRv2_det_infer.tar && cd ..


       wget -nc -P ./pretrain_models https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar
       cd ./pretrain_models && tar xf ch_ppocr_server_v2.0_det_train.tar && cd ..
       ```

   2. 导出onnx文件。

      1. 使用paddle2onnx工具导出onnx文件。

         在`ch_PP-OCRv2_det`工作目录下通过运行以下命令获取onnx模型。

         ```
         paddle2onnx \
             --model_dir ./inference/ch_PP-OCRv2_det_infer \
             --model_filename inference.pdmodel \
             --params_filename inference.pdiparams \
             --save_file ./ch_PP-OCRv2_det.onnx \
             --opset_version 11 \
             --enable_onnx_checker True \
             --input_shape_dict="{'x':[-1,3,-1,-1]}"
         ```

         参数说明请通过`paddle2onnx -h`命令查看。
         运行后在`ch_PP-OCRv2_det`目录下获得ch_PP-OCRv2_det.onnx文件。

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
             --model=./ch_PP-OCRv2_det.onnx \
             --output=./ch_PP-OCRv2_det_bs${batchsize} \
             --input_format=NCHW \
             --input_shape="x:${batchsize},3,-1,-1" \
             --soc_version=Ascend${chip_name} \
             --dynamic_image_size="736,736;736,800;736,960;736,992;736,1184;736,1248;736,1280;768,928;832,1536;992,736;1088,736;1184,736"
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
           运行成功后生成`ch_PP-OCRv2_det_bs${batchsize}.om`模型文件。

2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。  


   b.  执行推理。

      ```
      python3 ch_PP-OCRv2_det_ais_infer.py \
          --ais_infer=${path_to_ais-infer}/ais_infer.py \
          --model=./ch_PP-OCRv2_det_bs${batchsize}.om \
          --inputs=./pre_data \
          --batchsize=${batchsize}
      ```

      -   参数说明：
           -   --ais_infer：ais_infer.py脚本路径
           -   --model：om模型路径。
           -   --inputs：输入数据集路径。
           -   --batchsize：om模型的batchsize。

      `${path_to_ais-infer}`为ais_infer.py脚本的存放路径。`${batchsize}`表示不同batch的om模型。。

      推理完成后结果保存在`ch_PP-OCRv2_det/results_bs${batchsize}`目录下。


   c.  精度验证。

      执行后处理脚本`ch_PP-OCRv2_det_postprocess.py`，参考命令如下：

      ```
      python3 ch_PP-OCRv2_det_postprocess.py \
          -c PaddleOCR/configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_cml.yml \
          -o Global.infer_img="./imgs/" Global.infer_results=${output_path}
      ```

      -   参数说明：

            -   -c：模型配置文件。
            -   -o：可选参数：Global.infer_img表示样本图片路径，Global.infer_results表示om推理结果路径。

      ${output_path}为推理结果的保存路径，命令执行完成后，每个推理结果对应的检测图片保存在`${output_path}/det_results/`目录下：

      在线推理命令如下：

      ```
      python3 PaddleOCR/tools/infer_det.py \
          -c PaddleOCR/configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_cml.yml \
          -o Global.infer_img="./imgs/" \
          Global.pretrained_model="./checkpoint/ch_PP-OCRv2_det_distill_train/best_accuracy"
      ```
      
      执行完成后在线推理结果保存在`ch_PP-OCRv2_det/output/det_db/det_results_Teacher/`目录下。

      可以将om后处理得到的样例图片的推理结果，与在线推理得到的样例图片的推理结果进行对比，观察文本检测框的效果，来验证om的推理精度。

   d.  性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench \
          --model=./ch_PP-OCRv2_det_bs${batchsize}.om \
          --loop=50 \
          --dymHW=768,928 \
          --batchsize=${batchsize}
      ```

      -   参数说明：

          -   --model：om模型路径。
          -   --loop：推理次数。
          -   --dymHW：动态分辨率参数，指定模型输入的实际H、W。
          -   --batchsize：om模型的batch。

      `${batchsize}`表示不同batch的om模型。

      纯推理完成后，在ais_bench的屏显日志中`throughput`为计算的模型推理性能。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------- | ---------- | ---------- | --------------- |
|Ascend310P3| 1          | 样例图片 | 见备注 | 401.416 fps |
|Ascend310P3| 4          | 样例图片 | 见备注 | 302.236 fps |
|Ascend310P3| 8          | 样例图片 | 见备注 | 295.902 fps |
|Ascend310P3| 16         | 样例图片 | 见备注 | 288.953 fps |
|Ascend310P3| 32         | 样例图片 | 见备注 | 286.025 fps |
|Ascend310P3| 64         | 样例图片 | 见备注 | 287.518 fps |

   - 备注：可以将om后处理得到的样例图片的推理结果，与在线推理得到的样例图片的推理结果进行对比，观察文本检测框的效果，来验证om的推理精度。