# SSD模型-推理指导

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

SSD将detection转化为regression的思路，可以一次完成目标定位与分类。该算法基于Faster RCNN中的Anchor，提出了相似的Prior box；该算法修改了传统的SSD网络：将SSD的FC6和FC7层转化为卷积层，去掉所有的Dropout层和FC8层。同时加入基于特征金字塔的检测方式，在不同感受野的feature map上预测目标。

- 参考实现：

  ```shell
  url=https://github.com/open-mmlab/mmdetection.git
  branch=master
  commit_id=a21eb25535f31634cef332b09fc27d28956fb24b
  model_name=ssd
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 300 x 300 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小                  | 数据排布格式 |
  | -------- | -------- | --------------------- | ------------ |
  | boxes    | FLOAT32  | batchsize x 8732 x 4  | ND           |
  | labels   | FLOAT32  | batchsize x 8732 x 80 | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                            | 版本    | 环境准备指导                                                                                          |
  | --------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                      | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 5.1.RC2 | -                                                                                                     |
  | Python                                                          | 3.7.13  | -                                                                                                     |
  | PyTorch                                                         | 1.9.0   | -                                                                                                     |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

- 该模型需要以下依赖

  **表 1**  依赖列表

  | 依赖名称        | 版本     |
  | --------------- | -------- |
  | torch           | 1.8.1    |
  | torchvision     | 0.9.1    |
  | onnx            | 1.7.0    |
  | onnxruntime     | 1.12.0   |
  | numpy           | 1.21.6   |
  | Opencv-python   | 4.2.0.34 |
  | mmpypycocotools | 12.0.3   |
  | mmcv-full       | 1.2.7    |
  | mmdet           | 2.8.0    |
  | protobuf        | 3.20.0   |
  | decorator       | \        |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码

1. 获取SSD源代码并修改mmdetection。
   ```shell
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   git reset --hard a21eb25535f31634cef332b09fc27d28956fb24b
   patch -p1 < ../ssd_mmdet.diff
   pip install -v -e .
   cd ..
   ```

2. 安装依赖。
   ```shell
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   推理数据集采用 [coco_val_2017](http://images.cocodataset.org)，数据集下载后存放路径：`dataset=/root/datasets/coco`

   目录结构：

   ```
   ├── coco
   │    ├── val2017
   │    ├── annotations
   │         ├──instances_val2017.json
   ```

2. 数据预处理。

   将原始数据集转换为模型输入的二进制数据。执行 `ssd_preprocess.py` 脚本。

   ```shell
   python ssd_preprocess.py \
          --image_folder_path $dataset/val2017 \
          --bin_folder_path val2017_ssd_bin
   ```

   - 参数说明：

      -   --image_folder_path：原始数据验证集（.jpg）所在路径。
      -   --bin_folder_path：输出的二进制文件（.bin）所在路径。

   每个图像对应生成一个二进制文件。

3. 生成数据集info文件。

   运行 `get_info.py` 脚本，生成图片数据info文件。
   ```shell
   python get_info.py jpg $dataset/val2017 coco2017_ssd_jpg.info
   ```

   - 参数说明：

      -   第一个参数：生成的数据集文件格式。
      -   第二个参数：预处理后的数据文件相对路径。
      -   第三个参数：生成的info文件名。

   运行成功后，在当前目录中生成 `coco2017_ssd_jpg.info`。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      获取经过训练的权重文件：[ssd300_coco_20200307-a92d2092.pth](http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth)

   2. 导出onnx文件。

      使用pytorch2onnx.py导出onnx文件。

      ```shell
      python mmdetection/tools/pytorch2onnx.py \
              mmdetection/configs/ssd/ssd300_coco.py \
              ./ssd300_coco_20200307-a92d2092.pth \
              --output-file=ssd300_coco_dynamic_bs.onnx \
              --shape=300 \
              --mean 123.675 116.28 103.53 \
              --std 1 1 1
      ```

      - 参数说明：

         -   --output-file：为ONNX模型文件。
         -   --shape：输入的图片大小。
         -   --mean：输入数据预处理均值。
         -   --std：输入数据预处理方差。

      获得 `ssd300_coco_dynamic_bs.onnx` 文件。

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
         batchsize=8    # 以8batch为例
         atc --model=ssd300_coco_dynamic_bs.onnx \
              --framework=5 \
              --output=ssd300_coco_bs8 \
              --input_format=NCHW \
              --input_shape="input:${batchsize},3,300,300" \
              --log=error \
              --soc_version=Ascend${chip_name} \
              --buffer_optimize=off_optimize \
              --precision_mode=allow_fp32_to_fp16
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --buffer_optimize：
           -   --precision_mode：

           运行成功后生成 `ssd300_coco_bs8.om` 模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。
      ```shell
      python -m ais_bench \
              --model ssd300_coco_bs8.om \
              --input ./val2017_ssd_bin \
              --batchsize 8 \
              --output out
      ```

      - 参数说明：

         -   --model：为.OM模型文件的路径。
         -   --input：转换之后的二进制数据集路径。
         -   --batchsize：batch维度大小，与输入的.OM模型文件的batch维度一致。
         -    --output：模型推理结果存放的路径。

      推理后的输出在 `--output` 所指定目录下。


   3. 精度验证。

      调用coco_eval.py评测map精度：

      ```shell
      det_path=postprocess_out
      python ssd_postprocess.py \
              --bin_data_path=out/2022_*/ \
              --score_threshold=0.02 \
              --test_annotation=coco2017_ssd_jpg.info \
              --nms_pre 200 \
              --det_results_path ${det_path}
      python txt_to_json.py --npu_txt_path ${det_path}
      python coco_eval.py --ground_truth /root/datasets/coco/annotations/instances_val2017.json
      ```

      - 参数说明：

         -   --bin_data_path：为推理结果存放的路径。
         -   --score_threshold：得分阈值。
         -   --test_annotation：原始图片信息文件。
         -   --nms_pre：每张图片获取框数量的阈值。
         -   --det_results_path：后处理输出路径。
         -   --npu_txt_path：后处理输出路径。
         -   --ground_truth：instances_val2017.json文件路径。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

|           | mAP      |
| --------- | -------- |
| 310精度   | mAP=25.4 |
| 310P3精度 | mAP=25.4 |
| 310B1精度 | mAP=25.4 |


| Throughput | 310*4    | 310P3    | 310B1 |
| ---------- | -------- | -------- | ----- |
| bs1        | 179.194  | 298.5514 | 75.42 |
| bs4        | 207.596  | 337.0112 | 77.9  |
| bs8        | 211.7312 | 323.5662 | 79.77 |
| bs16       | 211.288  | 318.1392 | 77.84 |
| bs32       | 200.2948 | 318.7303 | 79.78 |
| bs64       | 196.4192 | 313.0790 | 48.36 |
| 最优batch  | 211.7312 | 337.0112 | 79.77 |