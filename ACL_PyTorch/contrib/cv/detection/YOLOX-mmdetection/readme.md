# YOLOX-mmdetection模型-推理指导

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
  url=https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox
  commit_id=6b87ac22b8d9dea8cc28b9ce84909e6c311e6268
  code_path=ACL_PyTorch/contrib/cv/detection/YOLOX-mmdetection
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

- 该模型需要以下依赖

  **表 2**  依赖列表

  | 依赖名称      | 版本     |
  | ------------- | -------- |
  | onnx          | 1.7.0    |
  | torchvision   | 0.8.0    |
  | opencv-python | 4.5.5.64 |
  | sympy         | 1.9      |
  | cython        | 0.29.28  |
  | mmcv-full     | 1.4.6    |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```shell
   git clone -b master https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   git reset 6b87ac22b8d9dea8cc28b9ce84909e6c311e6268 --hard
   patch -p1 < ../YOLOX.patch
   pip install -v -e .  # or  python3 setup.py develop
   cd ..
   pip install mmcv-full==1.4.6 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8/index.html
   ```

2. 安装依赖。

   ```shell
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   请参考开源代码仓方式获得[COCO2017数据集](https://cocodataset.org/)，并根据需要置于服务器上（如 `dataset=/root/dataset/coco`），val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json，文件目录结构如下：

   ```
    root
    ├── dataset
    │   ├── coco
    │   │   ├── annotations
    │   │   ├── val2017
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行YOLOX_preprocess.py脚本，完成预处理。

   ```shell
   python YOLOX_preprocess.py --image_src_path ${dataset}/val2017 \
                              --bin_file_path val2017_bin \
                              --meta_file_path val2017_bin_meta
   ```

   + 参数说明：
     + image_src_path：数据集路径
     + --bin_file_path：图像对应生成的二进制bin文件
     + --mate_file_path：图像对应生成的附加信息文件

   每个图像对应生成一个二进制bin文件，一个附加信息文件，文件分别保存在val2017_bin与val2017_bin_meta文件夹

3. 生成数据集info文件。

   生成数据集info文件，执行gen_dataset_info.py，会生成yolox_meta.info用于后处理。

   ```shell
   python3 gen_dataset_info.py \
            ${dataset} \
            mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py \
            val2017_bin  \
            val2017_bin_meta \
            yolox_meta.info \
            640 640
   ```

   + 参数说明
     + ${dataset}：数据集路径。
     + mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py：模型配置文件，包含在在开源库中。
     + val2017_bin：预处理后的数据文件的相对路径。
     + val2017_bin_meta：预处理后的数据文件的相对路径。
     + yolox_meta.info：生成的数据集文件保存的路径。
     + 640：图片宽。
     + 640：图片高。

   运行成功后，在当前目录中生成“yolox_meta.info”。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       我们利用官方的PTH文件进行验证，官方PTH文件可从原始开源库中获取，我们需要[yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth)文件,请将其放在与readme.md文件同一目录内。

   2. 导出onnx文件。

      ```shell
      cd mmdetection
      python3 tools/deployment/pytorch2onnx.py \
                  configs/yolox/yolox_x_8x8_300e_coco.py \
                  ../yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth \
                  --output-file ../yolox.onnx \
                  --shape 640 640 \
                  --dynamic-export
      cd ..
      ```

      获得yolox.onnx文件。

      + 参数说明
         + configs/yolox/yolox_x_8x8_300e_coco.py：使用的开源代码文件路径。
         + ../yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth：权重文件名称。
         + --output-file：输出文件名称。
         + --shape：图片参数。

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
          atc --framework=5 \
               --model=yolox.onnx \
               --output=yolox_bs1 \
               --input_format=NCHW \
               --input_shape="input:1,3,640,640" \
               --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。


      运行成功后生成yolox_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```shell
        python3 -m ais_bench \
                  --model=yolox_bs1.om \
                  --input=val2017_bin \
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

        >**说明：**
        >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见--help命令。

   3. 精度验证。

      ```shell
      python3 YOLOX_postprocess.py --dataset_path ${dataset} \
                                    --model_config mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py \
                                    --bin_data_path ./bs1/ \
                                    --meta_info_path yolox_meta.info \
                                    --num_classes 81
      ```

      - 参数说明：

        - --dataset_path：数据集路径
        - --model_config：模型配置文件路径
        - --bin_data_path：推理结果所在路径
        -  --meta_info_path：gen_dataset_info.py生成的后处理文件
        -  --num_classes：目标检测类别数

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| Precision |                |
| --------- | -------------- |
| 标杆精度  | Box   AP:50.9% |
| 310P3精度 | Box   AP:51.0% |

| 芯片型号 | Batch Size | 数据集   | 性能   |
| -------- | ---------- | -------- | ------ |
| 310P3    | 1          | coco2017 | 41.324 |
| 310P3    | 4          | coco2017 | 56.24  |
| 310P3    | 8          | coco2017 | 67.41  |
| 310P3    | 16         | coco2017 | 43.76  |
| 310P3    | 32         | coco2017 | 55.99  |
| 310P3    | 64         | coco2017 | 77.49  |