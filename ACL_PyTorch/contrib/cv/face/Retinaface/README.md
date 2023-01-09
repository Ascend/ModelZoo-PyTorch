# RetinaFace模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Retinaface模型是2019年提出的人脸检测模型，使用deformable convolution和dense regression loss，当时在WiderFace数据集上达到SOTA。本文档使用的模型是基于mobilenet（0.25）结构的轻量版本，基于RetinaNet的结构，采用特征金字塔技术，实现了多尺度信息的融合，对检测小物体有重要作用。


- 参考实现：

  ```
  url=https://github.com/biubug6/Pytorch_Retinaface
  commit_id=b984b4b775b2c4dced95c1eadd195a5c7d32a60b
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                          | 数据排布格式 |
  | -------- |-----------------------------| ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 1000 x 1000 | NCHW         |


- 输出数据

  | 输出数据    | 数据类型  | 大小                     | 数据排布格式  |
  |---------|------------------------|----------|------------| 
  | output0 | FLOAT32 | batchsize x 41236 x 4  | ND         |
  | output1 | FLOAT32 | batchsize x 41236 x 2  | ND         |
  | output2 | FLOAT32 | batchsize x 41236 x 10 | ND         |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动 

  **表 1**  版本配套表

  | 配套                                                         | 版本      | 环境准备指导                                                 |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>
1. 获取源码。

   ```
   git clone https://github.com/biubug6/Pytorch_Retinaface
   cd Pytorch_Retinaface
   git reset b984b4b775b2c4dced95c1eadd195a5c7d32a60b --hard
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用[WiderFace](http://shuoyang1213.me/WIDERFACE/index.html)
   的3226张验证集进行测试。获取数据集并解压后将images文件夹放在/data/widerface/val文件夹下。目录结构如下：

   ```
   Retinaface
   ├── data
      ├── widerface
         ├── val
            ├── images
               ├── 0-Parade
               ├── 1-Handshaking
               ├── ...
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行预处理脚本 retinaface_pth_preprocess.py，完成预处理。

   ```
   python3 retinaface_pth_preprocess.py
   ```

   运行成功后，生成的二进制文件默认放在./widerface文件夹下。目录结构如下：
      ```
   Retinaface
      ├── widerface
         ├── prep
         └── prep_info
      ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       [Retinaface基于mobilenet0.25的预训练权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Retinaface/PTH/mobilenet0.25_Final.pth)

   2. 导出onnx文件。

      1. 使用pth2onnx.py导出onnx文件。

         运行pth2onnx.py脚本。

         ```
         python3 pth2onnx.py -m mobilenet0.25_Final.pth
         ```

         获得retinaface.onnx文件。

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
         #以batch size=16为例
         atc --model retinaface.onnx \
             --framework 5 \
             --output retinaface_bs16 \
             --input_shape "image:16,3,1000,1000" \
             --soc_version Ascend${chip_name} \
             --log error \
             --out_nodes "Concat_205:0;Softmax_206:0;Concat_155:0" \
             --enable_small_channel 1 \
             --insert_op_conf ./aipp.cfg
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --out\_nodes：onnx模型的输出节点，不同版本的torch和torchvision包可能会影响pth转换得到的onnx模型结构。如使用不同版本的包需要查看模型最后的output2，output1，output0对应的上一层节点名称。
           -   --enable\_small\_channel：是否使用small channel优化。
           -   --insert\_op\_conf=aipp.cfg:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。

           运行成功后生成retinaface_bs16.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
         #以batch size=16为例
         python3 -m ais_bench \
               --model retinaface_bs16.om \
               --batchsize 16 \
               --input ./widerface/prep/ \
               --output ./result 
               --outfmt BIN
        ```

        -   参数说明：

             -   model：om文件路径。
             -   input：输入文件路径。
             -   output：输出文件路径。
             -   outfmt：输出文件格式。

        推理后的输出默认在当前目录result下。


   3. 精度验证。
      1. 数据后处理

         ```
         python3 retinaface_pth_postprocess.py \
               --prediction-folder ./result/{timestamp} \
               --info-folder ./widerface/prep_info \
               --output-folder ./widerface_result
         ```

         - 参数说明：

         -  --prediction-folder：ais_infer工具的推理结果，默认为 ./result。{timestamp}。{timestamp} 表示 ais_infer 工具执行推理任务时的时间戳。
         -  --info-folder：验证集预处理时生成的info信息，默认为 ./widerface/prep_info
         -  --output-folder：处理结果的保存位置，默认为 ./widerface_result
         -  --confidence-threshold：置信度阈值，默认为 0.02

      2. 计算精度

         如果是第一次运行精度计算需要运行第二步，编译评估文件，之后运行可直接执行第三步中的精度计算
         ```
         cd Pytorch_Retinaface/widerface_evaluate
         python3 setup.py build_ext --inplace
         python3 evaluation.py -p ../../widerface_result/
         ```

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
      python3 -m ais_bench --model=${om_model_path} --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型文件路径。
        - --batchsize：模型对应的batch size。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|    310P   |    1     |   WiderFace     |  90.44%（Easy） 87.56%（Medium）  72.44%（Hard）   |       1078.13        |
|    310P   |    4     |   WiderFace     |       |       1134.78        |
|    310P   |    8     |   WiderFace     |       |       1025.77        |
|    310P   |    16     |   WiderFace     |   90.44%（Easy） 87.56%（Medium）  72.44%（Hard）  |       1502.58        |
|    310P   |    32     |   WiderFace     |      |       909.72        |
|    310P   |    64     |   WiderFace     |       |       894.14        |