# TOOD模型-推理指导


- [TOOD模型-推理指导](#tood模型-推理指导)
- [1. 概述](#1-概述)
  - [1.1 输入输出数据](#11-输入输出数据)
- [2. 推理环境准备](#2-推理环境准备)
- [3. 快速上手](#3-快速上手)
  - [3.1 获取源码](#31-获取源码)
  - [3.2 准备数据集](#32-准备数据集)
  - [3.3 模型推理](#33-模型推理)
- [4. 模型推理性能\&精度](#4-模型推理性能精度)

  

# <a name="1">1. 概述</a>

TOOD 是一种任务对齐的一阶段目标检测模型。 单阶段目标检测通常通过优化目标分类和定位两个子任务来实现，使用具有两个平行分支的头部，这可能会导致两个任务之间的预测出现一定程度的空间错位，TOOD 基于学习的方式显式地对齐这两个任务。


- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection.git
  branch=master
  commit_id=3b72b12fe9b14de906d1363982b9fba05e7d47c1
  model_name=tood
  ```


  	通过Git获取对应commit\_id的代码方法如下：

  ```bash
  git clone https://github.com/open-mmlab/mmdetection.git        # 克隆仓库的代码
  cd mmdetection/                                                # 切换到模型的代码仓目录
  git reset --hard 3b72b12fe9b14de906d1363982b9fba05e7d47c1      # 代码设置到对应的commit_id
  ```


## <a name="11">1.1 输入输出数据</a>

- 输入数据：HxW

  | 输入数据 | 数据类型 | 大小                  | 数据排布格式 |
  | -------- | -------- | --------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 1216 x 1216 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | dets     | (100, 5) | FLOAT32  | ND           |
  | labels   | (100, )  | INT64    | ND           |


# <a name="2">2. 推理环境准备</a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.8.0   | -                                                            |
| PyTorch                                                      | 1.8.1   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# <a name="3">3. 快速上手</a>

## <a name="31">3.1 获取源码</a>

1. 获取源码。

   ```bash
   git clone https://github.com/open-mmlab/mmdetection.git        # 克隆仓库的代码
   cd mmdetection/                                                # 切换到模型的代码仓目录
   git reset --hard 3b72b12fe9b14de906d1363982b9fba05e7d47c1      # 代码设置到对应的commit_id
   git apply TOOD.patch
   ```

2. 安装依赖。

   ```bash
   pip install -r requirement.txt
   ```

## <a name="32">3.2 准备数据集</a>

1. 获取原始数据集

   数据集名称：coco2017

    ```bash
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip val2017.zip
    unzip annotations_trainval2017.zip
    ```

   存放路径与目录结构：

   ```bash
   mmdetection
   |-- data
       |-- coco
           |-- annotations
           |   |-- instances_val2017.json
           |-- val2017
               |-- 000000000139.jpg
               |-- .....
               |-- .....
   ```

2. 数据预处理

   数据预处理将原始数据集转换为模型输入的数据。

   执行 tood_preprocess.py 脚本，完成预处理。

    ```bash
    python tood_preprocess.py \
        --image_src_path data/coco/val2017 \
        --bin_file_path data/coco/test \
        --height 1216 \
        --width 1216
    ```
   
    参数说明：
   
    - image_src_path：val2017 所在路径
    - bin_file_path：生成的  bin 文件夹的根目录
    -  height：预处理后图片高度
    -  width：预处理后图片宽度
   
    运行后生成的文件夹：data/coco/val2017_bin_1216_1216


## <a name="33">3.3 模型推理</a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      ```bash
      wget https://download.openmmlab.com/mmdetection/v2.0/tood/tood_r50_fpn_1x_coco/tood_r50_fpn_1x_coco_20211210_103425-20e20746.pth
      ```

   2. 导出onnx文件。

      1. 使用 tools/deployment/pytorch2onnx.py 导出onnx文件。

         运行 tools/deployment/pytorch2onnx.py 脚本。

         ```bash
         python tools/deployment/pytorch2onnx.py \
             configs/tood/tood_r50_fpn_1x_coco.py \
             ./tood_r50_fpn_1x_coco_20211210_103425-20e20746.pth \
             --output-file tood.onnx \
             --input-img  tests/data/color.jpg \
             --test-img data/coco/val2017/000000000139.jpg \
             --shape 1216 1216 \
             --show
         ```

         获得 tood.onnx 文件。

      2. 运行 tood_OnnxConvert.py

         onnx 模型中部分算子的输入类型不匹配，需要做转换

         ```bash
         python tood_OnnxConvert.py --input_name tood.onnx --output_name tood_convert.onnx
         ```

         参数说明：

         - input_name：输入 onnx 模型路径
         - output_name：输出转换后的 onnx 模型的路径

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```bash
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```bash
         npu-smi info
         #该设备芯片名为Ascend310P3
         回显如下：
         +--------------------------------------------------------------------------------------------+
         | npu-smi 22.0.0                       Version: 22.0.2                                       |
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 17.1         55                0    / 0              |
         | 0       0         | 0000:87:00.0    | 0            927  / 23054                            |
         +===================+=================+======================================================+
         ```

      3. 执行ATC命令。

         ```bash
         atc --framework=5 \
            --model=./tood_convert.onnx \
            --output=tood \
            --input_shape="input:1,3,1216,1216" \
            --log=error \
            --soc_version=${chip_name}
         ```

         参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

         运行成功后生成 tood.om 模型文件。

2. 开始推理验证。

    1.  使用ais_bench工具进行推理, ais_bench工具获取及使用方式请点击查看 [ais_bench推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer) 
   
   2.  执行推理

        ```bash
        mkdir output
        python -m ais_bench --model tood.om --input data/coco/val2017_bin_1216_1216 --output output 
        ```

        参数说明：
        
        - model：需要推理的 om 文件路径
        - input：数据集 bin 文件所在路径
        - output：推理结果保存的路径

        运行后会在 output 文件夹下面得到推理结果

        >**说明：** 
        >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见。

   3.  精度验证。

        调用 tood_postprecess.py 脚本与数据集标签 data/coco/annotations/instances_val2017.json 比对，可以获得Accuracy数据
   
        ```bash
        python tood_postprecess.py --ann_file_path data/coco/annotations/instances_val2017.json \
                            --bin_file_path output/2022_10_22-01_28_02 \
                            --height 1216 \
                            --width 1216
        ```
   
        参数说明：
        -  ann_file_path：instances_val2017.json 路径
        -  bin_file_path：推理结果路径
        -  height：预处理后图片高度
        -  width：预处理后图片宽度

   4.  性能验证。
   
       可使用 ais_bench 推理工具的纯推理模式验证不同 om 模型的性能，参考命令如下:

       ```bash
       python -m ais_bench --model tood.om --loop=20 --output ./xingneng/
       ```


# <a name="4">4. 模型推理性能&精度</a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 精度 | 性能 |
| ----------- | ---------- | -------- | ---- | ---- |
| Ascend310P3 | 1          | coco2017 | 42.2 | 14.6 |