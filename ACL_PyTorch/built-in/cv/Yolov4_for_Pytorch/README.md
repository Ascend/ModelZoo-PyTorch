
# YOLOV4模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

   - [输入输出数据](#ZH-CN_TOPIC_0000001126281702)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

YOLO是一个经典的物体检查网络，将物体检测作为回归问题求解。YOLO训练和推理均是在一个单独网络中进行。基于一个单独的end-to-end网络，输入图像经过一次inference，便能得到图像中所有物体的位置和其所属类别及相应的置信概率。YOLOv4在YOLOv3的基础上做了很多改进，其中包括近几年来最新的深度学习技巧，例如Swish、Mish激活函数，CutOut和CutMix数据增强方法，DropPath和DropBlock正则化方法，也提出了自己的创新，例如Mosaic（马赛克）和自对抗训练数据增强方法，提出了修改版本的SAM和PAN，跨Batch的批归一化（BN），共五大改进。所以说该文章工作非常扎实，也极具创新。

[Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao.YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv 2018.Thu, 23 Apr 2020 02:10:02 UTC (3,530 KB)](https://arxiv.org/abs/2004.10934)

- 参考实现：

  ```shell
  url=https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
  branch=master
  commit_id=78ed10cc51067f1a6bac9352831ef37a3f842784
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | images   | RGB_FP32 | batchsize x 3 x 416 x 416 | NCHW         |

- 输出数据

  | 输出数据    | 数据类型 | 大小       | 数据排布格式 |
  | ----------- | -------- | ---------- | ------------ |
  | Reshape_216 | FLOAT32  | 3x85x13x13 | NCHW         |
  | Reshape_203 | FLOAT32  | 3x85x26x26 | NCHW         |
  | Reshape_187 | FLOAT32  | 3x85x52x52 | NCHW         |

# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                            | 版本    | 环境准备指导                                                                                          |
| --------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
| 固件与驱动                                                      | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                            | 6.0.RC1 | -                                                                                                     |
| Python                                                          | 3.7.5   | -                                                                                                     |
| PyTorch                                                         | 1.8.0   | -                                                                                                     |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

    ```
    git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git
    cd pytorch-YOLOv4
    git reset --hard a65d219f9066bae4e12003bd7cdc04531860c672
    patch -p2 < ../yolov4.patch
    cd ..
    ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型需要coco2014数据集，数据集下载[地址](https://cocodataset.org/)

   数据集结构如下
   ```
    coco
     ├── annotations
     │   └── instances_minival2014.json
     └── images
         └── val2014

   ```

2. 生成数据集info文件。
   执行parse_json.py脚本。
   ```
   mkdir ground-truth
   python3 parse_json.py --dataset=./coco
   ```
   - 参数说明
      - ${dataset}：数据集路径。

   执行成功后，在当前目录下生成coco2014.name和coco_2014.info文件以及标签文件夹ground-truth。


3. 预处理数据集。
   ```
   python preprocess_yolov4_pytorch.py coco_2014.info yolov4_bin
   ```

## 模型推理<a name="section741711594517"></a>
1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      从 [官方仓库链接](https://github.com/Tianxiaomo/pytorch-YOLOv4) 中下载 `yolov4.pth`,放入pytorch-YOLOV4源码仓目录下。

   2. 导出onnx文件。

        ```shell
        cd pytorch-YOLOv4
        python demo_pytorch2onnx.py yolov4.pth data/dog.jpg -1 80 608 608
        cd ..
        mv pytorch-YOLOv4/yolov4_-1_3_608_608_dynamic.onnx .
        ```

        使用onnxsim简化模型
        ```
        python -m onnxsim yolov4_-1_3_608_608_dynamic.onnx yolov4_-1_3_608_608_dynamic.onnx
        ```
        
        获得 `yolov4_-1_3_608_608_dynamic.onnx` 文件。

        - 参数说明：

          - yolov4.pth：权重文件。
          - data/dog.jpg：样例图片。
          - -1 80 608 608：输入输出信息。

   3. 使用ATC工具将ONNX模型转OM模型。

       1. 配置环境变量。

          ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
          ```

       2. 执行命令查看芯片名称（$\{chip\_name\}）。

          ```shell
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
          atc --framework=5 \
               --model=yolov4_-1_3_608_608_dynamic.onnx \
               --output=yolov4_bs${batchsize} \
               --input_format=NCHW \
               --input_shape="input:${batchsize},3,608,608" \
               --log=error \
               --insert_op_conf=aipp.config \
               --enable_small_channel=1 \
               --soc_version=Ascend${chip_name} 
          ```

          - 参数说明：

            -   --model：为ONNX模型文件。
            -   --framework：5代表ONNX模型。
            -   --output：输出的OM模型。
            -   --input_format：输入数据的格式。
            -   --input_shape：输入数据的shape。
            -   --log：日志级别。
            -   --soc_version：处理器型号。

            运行成功后生成 `yolov4_bs${batchsize}.om` 模型文件。

2. 开始推理验证。

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。
      ```shell
      python3 -m ais_bench --model=yolov4_bs${bs}.om --input=./yolov4_bin --output=./ --output_dirname=result_bs${bs} --batchsize=${bs}
      ```

      -   参数说明：

          -   --model：om文件路径。
          -   --input:输入路径
          -   --output：输出路径。
          -   --output_dirname：输出文件保存路径。

         > **说明：**
         > 执行ais-infer工具请选择与运行环境架构相同的命令。

   3. 精度验证。

      使用bin_to_predict_yolov4_pytorch脚本输出特征图
      ```shell
      mkdir detection-results
      python3 bin_to_predict_yolov4_pytorch.py \
              --bin_data_path=./result_bs${bs}/ \
              --det_results_path=detection-results \
              --origin_jpg_path=./coco/images/ \
              --coco_class_names=coco2014.names
      ```
      -   参数说明：

          -   --bin_data_path：推理结果路径
          -   --det_results_path：特征图保存路径
          -   --origin_jpg_path：原始数据集图片路径
          -   --coco_class_names：数据集信息文件

      使用脚本map_calculate.py脚本计算输出特征图map值：
      ```shell
      python3 map_calculate.py --label_path=./ground-truth/ --npu_txt_path=./detection-results/
      ```
      -   参数说明：

          -   --label_path：标签路径
          -   --npu_txt_path：推理特征图路径



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
调用ACL接口推理计算，性能参考下列数据。

| Batch Size   | 数据集 | 精度 | 310P3 | 310B1 |
| ---------------- | ---------- | ---------- | --------------- | ---------------- |
|       1    |   coco2014         |     60.3%       |   152.80              |   42.38        |
|       4       |   coco2014        |            |           170.82      |      41.27      |
|       8       |    coco2014       |            |     171.15            |     42.33       |
|      16       |     coco2014      |    60.3%        |      170.97           |      29.63      |
|   32          |    coco2014      |            |     170.36            |     26.99       |
|   64          |    coco2014      |            |        167.28         |        27.19        |
|  |  | **最优性能** | **171.15** | **42.38** |

