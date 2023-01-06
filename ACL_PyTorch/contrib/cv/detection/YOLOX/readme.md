# YOLOX模型-推理指导

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

YOLOX是基于往年对YOLO系列众多改进而产生的目标检测模型，其采用无锚方式，并应用了解耦头和领先的标签分配策略 SimOTA.其在众多数据集中均获得了最佳结果。

- 参考实现：

  ```
  url=https://github.com/Megvii-BaseDetection/YOLOX
  commit_id=6880e3999eb5cf83037e1818ee63d589384587bd
  code_path=ACL_PyTorch/contrib/cv/detection/YOLOX
  model_name=YOLOX
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 640 x 640 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小    | 数据排布格式 |
  | -------- | -------- | ------- | ------------ |
  | output1  | FLOAT32  | 200 x 5 | NCHW         |
  | output2  | INT64    | 200     | NCHW         |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                            | 版本    | 环境准备指导                                                                                          |
  | --------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                      | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 5.1.RC2 | -                                                                                                     |
  | Python                                                          | 3.7.5   | -                                                                                                     |
  | PyTorch                                                         | 1.8.0   | -                                                                                                     |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```shell
   git clone https://github.com/Megvii-BaseDetection/YOLOX
   cd YOLOX
   git reset 6880e3999eb5cf83037e1818ee63d589384587bd --hard
   patch -p1 < ../Yolox-x.patch
   pip install -v -e .  # or  python3 setup.py develop
   cd ..
   ```

2. 安装依赖。

   ```shell
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   请参考开源代码仓方式获得[COCO2017数据集](https://cocodataset.org/)，并根据需要置于服务器上（如 `datasets_path=/root/dataset/coco`），val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json，文件目录结构如下：

   ```
    root
    ├── dataset
    │   ├── coco
    │   │   ├── annotations
    │   │   ├── val2017
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行Yolox_preprocess.py脚本，完成预处理。

   ```shell
   python Yolox_preprocess.py --dataroot ${datasets_path} --output './prep_data'
   ```

   + 参数说明：
     + dataroot：数据集路径
     + output：图像对应生成的二进制bin文件

   每个图像对应生成一个二进制bin文件，并保存在prep_data文件夹下

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       我们利用官方的PTH文件进行验证，官方PTH文件可从原始开源库中获取，我们需要[yolox_x.pth](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth)文件,请将其放在与readme.md文件同一目录内。

   2. 导出onnx文件。

      ```shell
      cd YOLOX
      python tools/export_onnx.py --output-name ../yolox_x.onnx -n yolox-x -c ../yolox_x.pth --no-onnxsim --dynamic
      cd ..
      ```

      获得yolox.onnx文件。

      + 参数说明
         + `output-name`：输出文件名称。
         + `-n`：模型名称。
         + `-c`：权重文件路径。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```shell
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
          # 这里仅以batchsize=1为例进行测试
          atc --model=yolox_x.onnx \
              --framework=5 \
              --output=yolox_x \
              --input_format=NCHW \
              --input_shape='images:1,3,640,640' \
              --log=error \
              --soc_version=Ascend${chip_name} \
              --out_nodes="Conv_498:0;Conv_499:0;Conv_491:0;Conv_519:0;Conv_520:0;Conv_512:0;Conv_540:0;Conv_541:0;Conv_533:0"
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。


      运行成功后生成yolox_x.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```shell
        python3 -m ais_bench \
                  --model=yolox_x.om \
                  --input=prep_data \
                  --output ./ \
                  --output_dirname=bs1 \
                  --outfmt BIN \
                  --batchsize 1
        ```

        -   参数说明：

             -   --model：om模型。
             -   --input：输入路径
             -   --output：输出路径。
             -   --outfmt：输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”。
             -   --ouyput_dirname:推理结果输出子文件夹。可选参数。与参数output搭配使用。
             -   --batchsize：模型batch size 默认为1 。

        推理后的输出默认在当前目录--output下。


   3. 精度验证。

      ```shell
      python Yolox_postprocess.py --dataroot ${datasets_path} --dump_dir 'bs1'
      ```

      - 参数说明：

        - --dataroot：数据集路径
        - --dump_dir：`ais-infer`推理结果文件目录

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| Precision |                |
| --------- | -------------- |
| 标杆精度  | Box   AP:51.2% |
| 310P3精度 | Box   AP:51.2% |

| 芯片型号 | Batch Size | 数据集   | 性能   |
| -------- | ---------- | -------- | ------ |
| 310P3    | 1          | coco2017 | 41.324 |
| 310P3    | 4          | coco2017 | 56.24  |
| 310P3    | 8          | coco2017 | 67.41  |
| 310P3    | 16         | coco2017 | 43.76  |
| 310P3    | 32         | coco2017 | 55.99  |
| 310P3    | 64         | coco2017 | 77.49  |