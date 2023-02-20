# YolactEdge模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

YolactEdge模型是一个边缘设备上的实时实例分割模型。YolactEdge基于YOLACT方法进行了两项改进：优化TensorRT并权衡速度和准确性，利用视频中时间冗余的新型特征。优化TensorRT的方法为压缩模型的权重数据的精度，提升了在图片上的运行速度。对于视频流对象，该模型重复利用backbone获取到的特征信息以减少计算量，提升运行速度。


- 参考实现：

  ```
  url=https://github.com/haotian-liu/yolact_edge
  commit_id=a9a00281b33b3ac90253a4939773308a8f95e21d
  ```
  


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 550 x 550 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output0  | FLOAT32  | batchsize x 256 x 69 x 69 | NCHW         |
  | output1  | FLOAT32  | batchsize x 256 x 35 x 35 | NCHW         |
  | output2  | FLOAT32  | batchsize x 256 x 18 x 18 | NCHW         |
  | output3  | FLOAT32  | batchsize x 256 x 9 x 9 | NCHW         |
  | output4  | FLOAT32  | batchsize x 256 x 5 x 5 | NCHW         |
  | output5  | FLOAT32  | batchsize x 138 x 138 x 32 | NCHW         |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.10.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/haotian-liu/yolact_edge.git
   cd yolact_edge
   git reset a9a00281b33b3ac90253a4939773308a8f95e21d --hard
   ```
   clone源码仓后，需要将pth2onnx.py、yolact_edge.patch、preprocess.py、postprocess.py、requirements.txt文件拷贝到源码仓内。目录结构如下：
   ```
   yolact_edge
   ├── pth2onnx.py
   ├── yolact_edge.patch
   ├── preprocess.py
   ├── postprocess.py
   ├── requirements.txt
   ├── data
   ├── ...
   ```

   运行补丁文件，修改源码。
   ```
   git apply yolact_edge.patch
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型采用COCO2017数据集的验证集进行精度评估。获取数据集并解压后将coco文件夹放在data文件夹下。目录结构如下：

   ```
   yolact_edge
   ├── data
      ├── coco
         ├── annotations
         └── val2017
      ├── scripts
      ├── yolact_edge_example_1.gif
      ├── yolact_edge_example_2.gif
      └── yolact_edge_example_3.gif
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行preprocess.py脚本，完成预处理。

   ```
   python3 preprocess.py --image_path ./data/coco/val2017 --json_file ./data/coco/annotations/instances_val2017.json --save_path ./inputs
   ```
   - 参数说明
      - image_path：图片所在文件夹，默认为 ./data/coco/val2017
      - json_file：json文件所在路径，默认为 ./data/coco/annotations/instances_val2017.json
      - save_path：生成的bin文件存储路径，默认为 ./inputs

   运行成功后，生成的二进制文件和数据集索引文件默认放在./inputs文件夹下。目录结构如下：
   
   ```
   yolact_edge
   ├── inputs
         ├── bin_file
         └── ids.json
   ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [YolactEdge权重下载地址](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/YOLACTEDGE/PTH/yolact_edge_resnet50_54_800000.pth)


   2. 导出onnx文件。

      1. 使用pth2onnx.py导出onnx文件。

         运行pth2onnx.py脚本。

         ```
         python3 pth2onnx.py \
            --config=yolact_edge_resnet50_config \
            --trained_model=yolact_edge_resnet50_54_800000.pth
         ```
         - 参数说明：
            - trained_model：模型权重，默认为 yolact_edge_resnet50_54_800000.pth。
            - config：模型配置，默认为 yolact_edge_resnet50_config。

         获得yolact_edge.onnx文件。


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

         ```shell
         atc --model yolact_edge.onnx \
             --framework 5 \
             --output yolact_edge_bs${batch_size} \
             --log error \
             --soc_version Ascend${chip_name} \
             --input_shape "image:${batch_size},3,550,550"
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成`yolact_edge_bs${batch_size}.om`模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

      ```shell
      python3 -m ais_bench --model yolact_edge_bs${batch_size}.om --input inputs/bin_file/ --output ./ --output_dirname output
      ```

      -  参数说明：
         - model：om文件路径。
         - input：模型输入文件路径。
         - output：输出文件路径。

      推理后的输出默认在当前目录output下。


   3. 精度验证。

      调用脚本与数据集标签比对，可以获得mAP数据。

      ```
      python3 postprocess.py --file_path output \
            --trained_model yolact_edge_resnet50_54_800000.pth \
            --config yolact_edge_resnet50_config \
            --image_path ./data/coco/val2017 \
            --json_file ./data/coco/annotations/instances_val2017.json \
            --ids_path ./inputs/ids.json
      ```

      - 参数说明：

        - file_path：推理结果所在路径，默认为output。
        - trained_model：模型权重，默认为 yolact_edge_resnet50_54_800000.pth。
        - config：模型配置，默认为 yolact_edge_resnet50_config。
        - image_path：原始图片的存储路径，默认为 ./data/coco/val2017。
        - json_file：原始数据的标签文件，默认为 ./data/coco/annotations/instances_val2017.json
        - ids_path：预处理时数据集的索引文件，默认为 ./inputs/ids.json

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model=${om_model_path} --loop=100 --batchsize=${batch_size}
      ```

      - 参数说明：
        - --model：om模型文件路径。
        - --batchsize：模型输入对应的batch size。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度(mAP) | 性能 |
| --------- | ---------- | ---------- | -------- | ----------------- |
|    310P    |    1      |    COCO    |   27.96  |      270.35       |
|    310P    |    4      |    COCO    |          |      201.08       |
|    310P    |    8      |    COCO    |          |      175.41       |
|    310P    |    16     |    COCO    |          |      178.71       |
|    310P    |    32     |    COCO    |          |      183.77       |
|    310P    |    64     |    COCO    |          |      183.11       |