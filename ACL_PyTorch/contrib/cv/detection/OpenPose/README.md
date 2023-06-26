# OpenPose模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)


******




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

  OpenPose是基于卷积神经网络和监督学习的开源库，可以实现人的面部表情、躯干和四肢甚至手指的跟踪，不仅适用于单人也适用于多人，同时具有较好的鲁棒性。可以称是世界上第一个基于深度学习的实时多人二维姿态估计，是人机交互上的一个里程碑，为机器理解人提供了一个高质量的信息维度。
  
  与所有自底向上的方法类似，OpenPose管道由两部分组成：
  神经网络的推理提供两个张量:关键点热图及其成对关系。这个输出向下采样8次。
  按个人实例分组关键点。它包括将张量向上采样到原始图像大小，在热图峰值处提取关键点，并按实例进行分组。
  该网络首先提取特征，然后对热图和pafs进行初始估计，执行5个细化阶段。它可以找到18种类型的关键点。然后，分组过程从预定义的关键点对列表中搜索每个关键点的最佳配对，例如。左肘左腕，右臀右膝，左眼左耳等，共19对。


- 参考论文：[Daniil Osokin.Real-time 2D Multi-Person Pose Estimation on CPU:Lightweight OpenPose.(2018)](https://arxiv.org/abs/1811.12004)

- 参考实现：

  ```
  url=https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
  branch=master 
  commit_id=1590929b601535def07ead5522f05e5096c1b6ac
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 大小                      | 数据类型 | 数据排布格式 |
  | -------- | ------------------------- | -------- | ------------ |
  | input    | batchsize x 3 x 368 x 640 | RGB_FP32 | NCHW         |


- 输出数据

  | 输出数据 | 大小                     | 数据类型 | 数据排布格式 |
  | -------- | ------------------------ | -------- | ------------ |
  | output1  | batchsize x 19 x 46 x 80 | FLOAT32  | ND           |
  | output2  | batchsize x 38 x 46 x 80 | FLOAT32  | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套       | 版本    | 环境准备指导                                                 |
| ---------- | ------- | ------------------------------------------------------------ |
| 固件与驱动 | 1.0.16（NPU驱动固件版本为5.1.RC2） | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN       | 5.1.RC2 |                                                              |
| Python     | 3.7.5   |                                                              |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   <!--```-->
   <!--https://www.hiascend.com/zh/software/modelzoo/models/detail/1/c7f19abfe57146bd8ec494c0b377517c-->
   <!--```-->
    源码目录结构：

    ``` 
   ├── OpenPose_pth2onnx.py         //用于转换pth文件到onnx文件 
   ├── OpenPose_preprocess.py      //数据集预处理脚本，通过均值方差处理归一化图片并进行缩放填充等
   ├── OpenPose_postprocess.py     //验证推理结果脚本，模型输出的分类结果和标签，给出Accuracy 
   └── README.md
    ```

2. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git  
   cd lightweight-human-pose-estimation.pytorch
   git checkout master
   git reset --hard 1590929b601535def07ead5522f05e5096c1b6ac
   cd -
   ```

3. 安装依赖。

   ```
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型支持coco2017 5000张图片的验证集。请用户需自行获取coco val2017验证集，上传数据集到服务器任意目录并解压（如：当前文件夹/opt/npu/coco）。本模型将使用到val2017.zip验证集及annotations_trainval2017.zip中的person_keypoints_val2017.json数据标签。

   数据集文件夹结构如下：

    ```
   coco
   ├──val2017
       ├── img.jpg
   ├──annotations
       ├── captions_train2017.json
       ├── captions_val2017.json
       ├── instances_train2017.json
       ├── instances_val2017.json
       ├── person_keypoints_train2017.json
       ├── person_keypoints_val2017.json
    ```

2. 数据预处理。

   1. 建立数据存储文件夹。

      ```
      mkdir -p ./datasets/coco/prep_dataset
      mkdir ./output
      ```

   2. 数据预处理将原始数据集转换为模型输入的数据。

      执行“OpenPose_preprocess.py”脚本文件。

      ```
      python OpenPose_preprocess.py --src_path /opt/npu/coco/val2017 --save_path datasets/coco/prep_dataset --pad_txt_path output/pad.txt
      ```

      - 参数说明：
        - --src_path：为数据集路径。
        - --save_path：为模型输入数据存储路径。
        - --pad_txt_path：输出文件路径。

      执行成功后，在当前目录放入datasets/coco/prep_dataset下为预处理后的数据文件


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件[checkpoint_iter_370000.pth](https://pan.baidu.com/s/15BDVngC8XepdtlFZ5K8ZAw)，提取码k3w8，放在当前目录weights下。
        

   2. 导出onnx文件。

      使用checkpoint_iter_370000.pth导出onnx文件。

      运行OpemPose_pth2onnx.py脚本。

      ```
      python OpenPose_pth2onnx.py --checkpoint_path=./weights/checkpoint_iter_370000.pth --output_name=onnx2om/human-pose-estimation.onnx
      ```

      - 参数说明：
        - --checkpoint_path：权重pth文件路径。
        - --output_name：输出onnx文件路径。

      获得human-pose-estimation.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称(${chip_name})。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3
         回显如下：
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 17.6         57                0    / 0              |
         | 0       0         | 0000:3B:00.0    | 0            936 / 21534                            |
         +===================+=================+======================================================+
         ```

      3. 执行ATC命令。

         ```
         atc --framework=5 --model=onnx2om/human-pose-estimation.onnx --output=onnx2om/human-pose-estimation_bs1 --input_format=NCHW --input_shape="data:1,3,368,640" --log=debug --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

          运行成功后在onnx2om目录下生成 <u>human-pose-estimation_bs1.om</u> 模型文件。



2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  


   2. 创建输出的result文件夹。

      ```
      mkdir result
      ```


   3. 执行推理。

      执行命令

      ```
      python -m ais_bench --model onnx2om/human-pose-estimation_bs1.om --input datasets/coco/prep_dataset --output result --output_dirname dumpout_bs1 --outfmt TXT --batchsize 1
      ```

      -   参数说明：
          - --model：om模型的路径。
          - --input：输入模型的二进制文件路径。
          - --output：推理结果输出目录。
          - --output_dirname：推理结果输出的二级目录名。
          - --batchsize：输入数据的batchsize。

      推理后的输出在当前目录result下。


   4. 精度验证。

      调用OpenPose_postprocess.py脚本与数据集标签person_keypoints_val2017.json比对，可以获得Accuracy数据。

      ```
      python OpenPose_postprocess.py --dump_output_result_path result/dumpout_bs1/  --labels /opt/npu/coco/annotations/person_keypoints_val2017.json  --pad_txt_path ./output/pad.txt --detections_save_path ./output/result_b1.json
      ```

      -   参数说明：
           -   --dump_output_result_path：生成推理结果所在路径。
           -   --labels ：标签数据。
           -   --pad_txt_path：填充信息。
           -   --detections_save_path：生成结果文件。

   5. 性能验证

       可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，命令如下：

      ```
      python -m ais_bench --model=onnx2om/human-pose-estimation_bs1.om --loop=20 --batchsize=1
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>


| 芯片型号 | Batch Size | 数据集 | 精度   | 性能     |
| -------- | ---------- | ------ | ------ | -------- |
| 310P3    | 1          | coco2017 | 0.404 | 712.16  |
| 310P3    | 4          | coco2017 | 0.404 | 887.45  |
| 310P3    | 8          | coco2017 | 0.404 | 837.97 |
| 310P3    | 16         | coco2017 | 0.404 | 873.55 |
| 310P3    | 32         | coco2017 | 0.404 | 880.94 |
| 310P3    | 64         | coco2017 | 0.404 | 566.44 |