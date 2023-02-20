# AlphaPose模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

`AlphaPose` 是一个精准的多人姿态估计系统，是首个在 COCO 数据集上可达到 70+ mAP（72.3 mAP，高于 Mask-RCNN 8.2 个百分点），在 MPII 数据集上可达到 80+ mAP（82.1 mAP）的开源系统。为了能将同一个人的所有姿态关联起来，AlphaPose 还提供了一个称为 Pose Flow 的在线姿态跟踪器，这也是首个在 PoseTrack 挑战数据集上达到 60+ mAP（66.5 mAP）和 50+ MOTA（58.3 MOTA）的开源在线姿态跟踪器。

- 参考实现：

  ```
  url=https://github.com/MVIG-SJTU/AlphaPose.git
  branch=master
  commit_id=ddaf4b99327132f7617a768a75f7cb94870ed57c 
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | image    | FLOAT32  | batchsize x 3 x 256 x 192 | NCHW         |

- 输出数据

  | 输出数据 | 大小                      | 数据类型 | 数据排布格式 |
  | -------- | --------                  | -------- | ------------ |
  | output   | batch_size x 17 x 64 x 48 | FLOAT32  | NCHW         |

# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                            | 版本    | 环境准备指导                                                                                          |
| ------------------------------------------------------------    | ------- | ------------------------------------------------------------                                          |
| 固件与驱动                                                      | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                            | 6.0.RC1 | -                                                                                                     |
| Python                                                          | 3.7.5   | -                                                                                                     |
| PyTorch                                                         | 1.5.0+ | -                                                                                                     |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master         # 切换到对应分支
   cd ACL_PyTorch/contrib/cv/detection/AlphaPose              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   git clone https://github.com/cocodataset/cocoapi.git
   cd cocoapi/PythonAPI/ 
   make -j8
   python3.7 setup.py install
   cd -
   ```

3. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/MVIG-SJTU/AlphaPose.git ./AlphaPose
   cd AlphaPose
   git reset ddaf4b99327132f7617a768a75f7cb94870ed57c --hard
   git pull origin pull/592/head  # Ctrl-x退出
   patch -p1 < ../AlphaPose.patch
   python3.7 setup.py build develop
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。

   本模型采用 [coco_val2017](https://cocodataset.org/) ，解压到data文件夹下（如不存在则需要新建）

   数据目录结构请参考：

   ```
   data
   └── coco
    ├── annotations
    └── val2017
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。该模型预处理数据依赖模型推理，需要下载对应模型权重：

   获取[fast_res50_256x192.pth](https://drive.google.com/open?id=1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn)，在工程目录下新建文件夹models，将pth文件放置到models文件夹内：

   ```
   mkdir -p models
   mv fast_res50_256x192.pth models
   ```

   获取[yolov3-spp.weights](https://pan.baidu.com/s/1Zb2REEIk8tcahDa8KacPNA)放到对应文件夹下：

   ```
   mkdir -p ./AlphaPose/detector/yolo/data
   mv yolov3-spp.weights ./AlphaPose/detector/yolo/data
   ```

   执行预处理脚本，生成数据集预处理后的bin文件:

   ```
   python3 preprocess.py --dataroot ./data/coco --output ./prep_data --output_flip ./prep_data_flip
   ```

   - 参数说明：

     --dataroot: 数据集文件位置。

     --output：非flip输出文件位置。

     --output_flip：flip输出文件位置。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      相关权重文件已通过预处理步骤下载得到。

   2. 导出onnx文件。

      1. 使用脚本导出onnx文件。

         运行pth2onnx.py脚本。

         ```
         # pth转换为ONNX
         mkdir -p models
         python3 pth2onnx.py AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml models/fast_res50_256x192.pth models/fast_res50_256x192.onnx
         ```

         - 参数说明：第一个参数为模型配置文件，第二个参数是模型权重路径，第三个参数是导出onnx文件路径。

         获得models/fast_res50_256x192.onnx文件。

     2. 优化onnx。

        ```
        # 以bs1为例
        python3 -m onnxsim --input-shape="image:1,3,256,192" models/fast_res50_256x192.onnx models/fast_res50_256x192_bs1.onnx
        ```

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：**
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         回显如下：
         +-------------------|-----------------|------------------------------------------------------+
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
         # 以bs1为例
         atc --framework=5 --model=models/fast_res50_256x192_bs1.onnx --output=models/fast_res50_256x192_bs1 --input_format=NCHW --input_shape="image:1,3,256,192" --log=debug --soc_version=${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成模型文件models/fast_res50_256x192_bs1.om。

2. 开始推理验证。

   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
        # 以bs1为例,该模型需要同时推理两份数据，得到最佳结果
        mkdir -p results/bs1
        python3 -m ais_bench --model ./models/fast_res50_256x192_bs1.om --input ./prep_data --output ./results --output_dirname bs1 --batchsize 1
        mkdir -p results/bs1_flip
        python3 -m ais_bench --model ./models/fast_res50_256x192_bs1.om --input ./prep_data_flip --output ./results --output_dirname bs1_flip --batchsize 1
        ```
        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --output_dirname：保存目录名。
             -   --device：NPU设备编号。
             -   --outfmt: 输出数据格式。
             -   --batchsize：推理模型对应的batchsize。


        推理后的输出默认在当前目录outputs/bs1下。

   3.  精度验证。

      调用postprocess.py脚本与数据集标签比对，获得Accuracy数据。

      ```
      # 以bs1为例
      python3 postprocess.py --dataroot ./data/coco --dump_dir ./result/bs1 --dump_dir_flip ./result/bs1_flip
      ```

      -   参数说明：

        --dataroot：数据集所在路径。
        --dump_dir：非flip推理结果所在路径。
        --dump_dir_flip：flip推理结果所在路径。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度参考下列数据:

| 模型           | pth精度   | 310离线推理精度 | 310P离线推理精度 | 基准性能    | 310性能    | 310P性能   |
| :------:       | :------:  | :------:        | :------:         | :------:    | :------:   | :------:   |
| AlphaPose bs1  | mAP:71.73 | mAP:71.50       | mAP:71.47        | 627.502fps  | 330.596fps | 864.15fps  |
| AlphaPose bs16 | mAP:71.73 | mAP:71.50       | mAP:71.47        | 1238.543fps | 642.756fps | 1772.40fps |
| AlphaPose bs4  |           |                 |                  | 1082.605fps |            | 1641.56fps |
| AlphaPose bs8  |           |                 |                  | 1196.666fps |            | 1703.80fps |
| AlphaPose bs32 |           |                 |                  | 1400.707fps |            | 1412.49fps |
| AlphaPose bs64 |           |                 |                  | 1449.932fps |            | 1405.04fps |
