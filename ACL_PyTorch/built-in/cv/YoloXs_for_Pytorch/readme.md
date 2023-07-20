# YOLOXs模型-推理指导


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

YOLOX对YOLO系列进行了一些有经验的改进，将YOLO检测器转换为无锚方式，并进行其他先进的检测技术，即解耦头和领先的标签分配策略SimOTA，在大规模的模型范围内获得最先进的结果。


- 参考实现：

  ```
  url=https://github.com/Megvii-BaseDetection/YOLOX.git
  commit_id=c9d128384cf0758723804c23ab7e042dbf3c967f
  model_name=YoloXs
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | images   | FLOAT32  | batchsize x 3 x 640 x 640 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                    | 数据排布格式 |
  | -------- | -------- | ----------------------- | ------------ |
  | output   | FLOAT32  | batchsize x dim1 x dim2 | ND           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本      | 环境准备指导                                                 |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.4  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.3.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.7.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/Megvii-BaseDetection/YOLOX.git
   cd YOLOX
   git reset c9d128384cf0758723804c23ab7e042dbf3c967f --hard
   cd ..
   ```
   
2. 安装依赖

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持coco2017验证集。用户需自行获取数据集，将annotations文件和val2017文件夹解压并上传数据集到源码包路径下。目录结构如下：

   ```
   coco2017
   ├── annotations
   └── val2017 
   ```
   
2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行Yolox_preprocess.py脚本，将原始数据（.jpg）转化为二进制文件（.bin）。
   ```
   export PYTHONPATH=$PYTHONPATH:./YOLOX
   python3 Yolox_preprocess.py --dataroot=./coco2017 --output=prep_data
   ```
   
    - 参数说明：
      - --dataroot：原始数据集所在路径。
      - --output：输出的二进制文件所在路径。
   

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用Pytorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      yolox_s.pth权重文件[下载链接](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth)。

   2. 导出onnx文件。
      * 如果需要将nms以算子形式移至模型内，请参照graphmodify.md
      使用export_onnx.py导出onnx文件。
       
      ```
      python3 YOLOX/tools/export_onnx.py -c ./yolox_s.pth -f YOLOX/exps/default/yolox_s.py --dynamic
      ```
         
      获得yolox.onnx文件。

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
         atc --model=yolox.onnx --framework=5 --output=yolox_bs${batch_size} --input_format=NCHW --optypelist_for_implmode="Sigmoid" --op_select_implmode=high_performance --input_shape='images:${batch_size},3,640,640' --log=info --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --optypelist_for_implmode：设置optype列表中算子的实现模式。
           -   --op_select_implmode：设置网络模型中算子的实现模式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***yolox_bs${batch_size}.om***</u>模型文件。

2. 开始推理验证

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

      ```
      python3 -m ais_bench --model=yolox_bs${batch_size}.om --input=prep_data --output=result --output_dirname=bs${batch_size}  
      ```

      - 参数说明：

        -   --model：om文件路径。
        -   --input：输入数据目录。
        -   --output：推理结果输出路径。
        -   --output_dirname: 推理结果输出目录。

   3. 精度验证。

      调用Yolox_postprocess.py脚本与标签数据比对，可以获得精度数据。

      ```
      python3 Yolox_postprocess.py --dataroot=./coco2017 --dump_dir=result/bs${batch_size}
      ```

      - 参数说明：

        - --dataroot：原始数据集所在路径。
        - --dump_dir：推理结果生成路径。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
        python3 -m ais_bench --model=yolox_bs${batch_size}.om --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om文件路径。
        - --batchsize：batch大小

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度 | 性能  |
| -------- |------------| ------ | ---- |-----|
|     310P3     | 1          | coco2017 | map:0.401 | 626 |
|     310P3     | 4          | coco2017 | map:0.401 | 790 |
|     310P3     | 8          | coco2017 | map:0.401 | 591 |
|     310P3     | 16         | coco2017 | map:0.401 | 554 |
|     310P3     | 32         | coco2017 | map:0.401 | 550 |
|     310P3     | 64         | coco2017 | map:0.401 | 504 |