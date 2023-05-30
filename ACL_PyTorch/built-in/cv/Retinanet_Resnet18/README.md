# RetinaNet_Resnet18模型-推理指导

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

论文提出了一个简单、灵活、通用的损失函数Focal loss，用于解决单阶段目标检测网络检测精度不如双阶段网络的问题。这个损失函数是针对了难易样本训练和正负样本失衡所提出的，使得单阶段网络在运行快速的情况下，获得与双阶段检测网络相当的检测精度。此外作者还提出了一个Retinanet用于检验网络的有效性，其中使用Resnet和FPN用于提取多尺度的特征。

- 参考实现：

  ```shell
  url=https://github.com/open-mmlab/mmdetection
  commit_id=c14dd6c42efb63f662a63fe403198bac82f47aa6
  model_name=retinanet_r18_fpn_1x_coco
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小              | 数据排布格式 |
  | -------- | -------- | ----------------- | ------------ |
  | input    | RGB_FP32 | 1 x 3 x 1216 x 1216 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小    | 数据排布格式 |
  | -------- | -------- | ------- | ------------ |
  | dets     | FLOAT32  | 1 x 5 x 100 | ND           |
  | labels   | INT64    | 1 x 100 | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套       | 版本    | 环境准备指导                                                                                          |
  | ---------- | ------- | ----------------------------------------------------------------------------------------------------- |
  | 固件与驱动 | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN       | 6.3.RC1 | -                                                                                                     |
  | Python     | 3.7.5   | -                                                                                                     |
  | PyTorch    | 1.10.0   | -                                                                                                     |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```shell
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   git reset --hard c14dd6c42efb63f662a63fe403198bac82f47aa6
   cd ..
   ```

2. 安装依赖。

   ```shell
   pip3 install -r requirements.txt
   cd mmdetection
   pip3 install -v -e .
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型使用coco2017数据集的验证集。获取数据集并解压后将coco文件夹放在data文件夹下，文件目录结构如下：
   ```
    Retinanet
    ├── data
    │   ├── coco
    │   │   ├── annotations
    │   │   ├── val2017
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

	 运行retinanet_preprocess.py脚本，完成数据预处理。
   
	```shell
	python3 retinanet_preprocess.py --image_folder ./data/coco/val2017 --bin_folder_path input_data	
   	```

      - --image_folder: 原始数据验证集所在路径。
      - --bin_folder_path：输出的二进制文件（.bin）所在路径。

    运行成功后，会在当前目录下生成input_data文件夹。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       [Retinanet_r18_fpn_1x权重下载路径](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r18_fpn_1x_coco/retinanet_r18_fpn_1x_coco_20220407_171055-614fd399.pth)

   2. 导出onnx文件。

      1. 使用pytorch2onnx.py导出onnx文件。

         ```shell
         export PYTHONPATH=`pwd`/mmdetection:$PYTHONPATH
         python3 mmdetection/tools/deployment/pytorch2onnx.py mmdetection/configs/retinanet/retinanet_r18_fpn_1x_coco.py retinanet_r18_fpn_1x_coco_20220407_171055-614fd399.pth --shape 1216 --output-file retinanet.onnx
         ```
          - 参数说明：
            - 第一个参数为配置文件路径。
            - 第二个参数为权重文件路径。
            - --shape：输入数据大小。
            - --output-file：转出的onnx模型路径。

         运行成功后获得retinanet.onnx文件。

       2. 量化（可选）

         ```shell
         amct_onnx calibration --model retinanet.onnx --save_path ./amct --input_shape "input:1,3,1216,1216" --data_dir ./input_data/ --data_types "float32" --calibration_config amct.cfg
         mv Retinanet_Resnet18_result_deploy_model.onnx retinanet.onnx
         ```
         获得量化后的retinanet_int8.onnx

      3. 修改ONNX文件。

          请访问[auto-optimizer改图工具](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)代码仓，根据readme文档进行工具安装。
	 
          ```shell
          python3 -m auto_optimizer optimize retinanet.onnx retinanet_fix.onnx -k 4,8
          ```
          获得retinanet_fix.onnx文件。


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
         atc --model=retinanet_fix.onnx \
            --framework=5 \
            --output=retinanet \
            --log=error \
            --soc_version=Ascend${chip_name}
         ```

         - 参数说明：
            - --model: ONNX模型文件所在路径。
            - --framework: 5 代表ONNX模型。
            - --output: 生成OM模型的保存路径。
            - --log: 日志级别。
            - --soc_version: 处理器型号。

        运行成功后生成 `retinanet.om` 模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```shell
      python3 -m ais_bench \
            --model retinanet.om \
            --input ./input_data \
            --output ./ \
            --output_dirname output_data
      ```

      - 参数说明：
         - --model: OM模型路径。
         - --input: 存放预处理bin文件的目录路径。
         - --output: 存放推理结果的目录路径。
         - --output_dirname: 存放推理结果文件夹。

        推理后的输出默认在当前目录output_data下。


   3. 精度验证。
      - 执行后处理脚本，计算 mAP 精度：
         ```shell
         python3 retinanet_postprocess.py \
               --bin_data_path=./output_data/ \
               --test_annotation=data/coco/val2017/ \
               --ground_truth=data/coco/annotations/instances_val2017.json
         ```

         - 参数说明：
            - --bin_data_path: 推理结果所在路径
            - --test_annotation: 原始图片信息文件
            - --ground_truth: 数据集信息文件


   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证om模型的性能，参考命令如下：

      ```shell
      python3 -m ais_bench --model retinanet.om --loop 20
      ```

      -参数说明：
       + --model: om模型。
       + --loop: 循环次数。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
调用ACL接口推理计算，性能参考下列数据。
开源仓精度（mAP）：31.7

| 芯片型号 | Batch Size   | 数据集 | 精度  | 性能(FP16) | 性能(INT8) |
| --------- | ---------------- | ---------- | ---------- | --------------- | --------------- |
|  310P3    |         1        |   COCO2017 | mAP：31.6  |     16.78      |    18.32   |
该模型仅支持batch size为1

