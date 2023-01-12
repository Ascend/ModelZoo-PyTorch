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

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
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
   pip3 install -v -e .
   ```
   
2. 安装依赖

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   1. 本模型支持coco2017验证集。用户需自行获取数据集，将annotations文件和val2017文件夹解压并上传数据集到源码包路径下的dataset文件夹下。目录结构如下：

      ```
      ├── annotations
      └── val2017 
      ```
   

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   1. 在https://github.com/Megvii-BaseDetection/YOLOX 界面下载YOLOX-s对应的weights， 名称为yolox_s.pth

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

   3. 执行转模型命令。

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   cd YOLOX
   cp yolox_s.pth ./YOLOX
   cp -r YoloXs_for_Pytorch/* ./YOLOX
   bash test/pth2om.sh Ascend${chip_name} # Ascend310P3
   ```

2. 开始推理验证

   1. [获取benchmark工具](https://gitee.com/ascend/cann-benchmark/tree/master/infer)

      将benchmark.x86_64或benchmark.aarch64放到当前YoloXs_for_Pytorch目录下

   2. 执行推理。

        1. 精度

        ```
        bash test/eval-acc.sh --datasets_path=/root/datasets  
        ```

        结果保存在results.txt文件中

        2. 性能测试：

        可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
        python3 -m ais_bench --model=${om_model_path} --loop=1000 --batchsize=${batch_size}
        ```

        - 参数说明：
          - --model：om模型
          - --batchsize：模型batchsize
          - --loop: 循环次数

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度 | 310P性能 |
| -------- | ---------- | ------ | ---- | ---- |
|     310P3     |   4   | coco2017 | map:0.401 |  890fps  |