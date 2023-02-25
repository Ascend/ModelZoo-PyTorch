# HigherHRNet模型-推理指导

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

自下而上的人体姿态估计方法由于尺度变化的挑战而难以为小人体预测正确的姿态。HigherHRNet：一种新的自下而上的人体姿势估计方法，用于使用高分辨率特征金字塔学习尺度感知表示。该方法配备了用于训练的多分辨率监督和用于推理的多分辨率聚合，能够解决自下而上的多人姿势估计中的尺度变化挑战，并能更精确地定位关键点，尤其是对于小人物。 HigherHRNet中的特征金字塔包括HRNet的特征图输出和通过转置卷积进行上采样的高分辨率输出。

- 参考实现：

  ```shell
  url=https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation.git
  commit_id=aa23881492ff511185acf756a2e14725cc4ab4d7
  code_path=ACL_PyTorch/contrib/cv/pose_estimation/HigherHRNet
  model_name=HigherHRNet
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32  | batchsize x 3 x 512x 512  | NCHW         |
  | input    | FLOAT32  | batchsize x 3 x 512x 576  | NCHW         |
  | input    | FLOAT32  | batchsize x 3 x 512x 640  | NCHW         |
  | input    | FLOAT32  | batchsize x 3 x 512x 704  | NCHW         |
  | input    | FLOAT32  | batchsize x 3 x 512x 768  | NCHW         |
  | input    | FLOAT32  | batchsize x 3 x 512x 832  | NCHW         |
  | input    | FLOAT32  | batchsize x 3 x 512x 896  | NCHW         |
  | input    | FLOAT32  | batchsize x 3 x 512x 960  | NCHW         |
  | input    | FLOAT32  | batchsize x 3 x 512x 1024 | NCHW         |
  | input    | FLOAT32  | batchsize x 3 x 576x 512  | NCHW         |
  | input    | FLOAT32  | batchsize x 3 x 640x 512  | NCHW         |
  | input    | FLOAT32  | batchsize x 3 x 704x 512  | NCHW         |
  | input    | FLOAT32  | batchsize x 3 x 768x 512  | NCHW         |
  | input    | FLOAT32  | batchsize x 3 x 832x 512  | NCHW         |
  | input    | FLOAT32  | batchsize x 3 x 896x 512  | NCHW         |
  | input    | FLOAT32  | batchsize x 3 x 960x 512  | NCHW         |
  | input    | FLOAT32  | batchsize x 3 x 1024x 512 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小                               | 数据排布格式 |
  | -------- | -------- | ---------------------------------- | ------------ |
  | output1  | FLOAT32  | batchsize x 17 x h/4 x w/41 x 1000 | NCHW         |
  | output2  | FLOAT32  | batchsize x 17 x h/2 x w/2         | NCHW         |

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
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd cocoapi/PythonAPI
   make install
   cd ../..
   git clone https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
   cd HigherHRNet-Human-Pose-Estimation
   patch -p1 < ../HigherHRNet.patch
   cd ..
   ```

2. 安装依赖。

   ```shell
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型支持COCO2017验证集。请用户需自行获取[COCO2017](https://cocodataset.org/)数据集，上传数据集到本项目路径下。目录结构如下：
   > 因为模型代码开源仓配置文件限制，请注意数据集配置路径

   ```
    data
     |-- coco
      `-- |-- annotations
          |   |-- person_keypoints_train2017.json
          |   `-- person_keypoints_val2017.json
          `-- images
              |-- train2017
              |   |-- 000000000009.jpg
              |   |-- 000000000025.jpg
              |   |-- 000000000030.jpg
              |   |-- ...
              `-- val2017
                  |-- 000000000139.jpg
                  |-- 000000000285.jpg
                  |-- 000000000632.jpg
                  |-- ...
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行 `HigherHRNet_preprocess.py` 脚本，完成预处理。

   ```shell
   python3 HigherHRNet_preprocess.py --output prep_dir --output_flip prep_flip_dir
   ```

   + 参数说明：
     + --output：输出的二进制文件（.bin）所在路径。
     + --output_flip：输出的二进制文件flip（.bin）所在路径。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       获取权重文件 [pose_higher_hrnet_w32_512.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/HigherHRNet/PTH/pose_higher_hrnet_w32_512.pth)

   2. 导出onnx文件。

      使用HigherHRNet_pth2onnx.py导出onnx文件。

      ```shell
      python3 HigherHRNet_pth2onnx.py \
              --weights pose_higher_hrnet_w32_512.pth \
              --onnx_path  higher_hrnet_dynamic.onnx
      ```

      - 参数说明：
        - --weights：为pth模型文件输入。
        - --onnx_path：onnx文件输出。

      获得higher_hrnet_dynamic.onnx文件。

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
             --model=higher_hrnet_dynamic.onnx \
             --output=higher_hrnet_dynamic \
             --input_format=ND \
             --input_shape="input:1,3,-1,-1" \
             --dynamic_dims="1024,512;960,512;896,512;832,512;768,512;704,512;640,512;576,512;512,512;512,576;512,640;512,704;512,768;512,832;512,896;512,960;512,1024" \
             --out_nodes="Conv_770:0;Conv_795:0"\
             --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --dynamic_dims：图片的动态分辨率参数。
           -   --soc\_version：处理器型号。

         运行成功后生成pose_higher_hrnet_dynamic.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

      ```shell
         python3 -m ais_bench \
                --model=higher_hrnet_dynamic.om \
                --input=./prep_dir \
                --output=./ --output_dirname=output_dir \
                --outfmt NPY \
                --auto_set_dymdims_mode 1
         python3 -m ais_bench \
                --model=higher_hrnet_dynamic.om \
                --input=./prep_flip_dir \
                --output=./ --output_dirname=output_flip_dir \
                --outfmt NPY \
                --auto_set_dymdims_mode 1
      ```

      -  参数说明：

         - model：om模型。
         - input：模型需要的输入。
         - output：推理结果输出路径。
         - outfmt：输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”。
         - output_dirname:推理结果输出子文件夹。可选参数。与参数output搭配使用。
         - auto_set_dymdims_mode：自动匹配输入数据的shape。

        HigherHRNet中的特征金字塔包括HRNet的特征图输出和通过转置卷积进行上采样的高分辨率输出,其中output_dir是特征图输出的推理结果，output_flip_dir是高分辨率输出的推理结果。


   3. 精度验证。

      ```shell
      python3 HigherHRNet_postprocess.py  --dump_dir ./output_dir --dump_dir_flip ./output_flip_dir
      ```

      - 参数说明：

        - --dump_dir：生成特征图推理结果所在路径。
        - --dump_dir_flip：生成高分辨率推理结果所在路径。

      后处理输出的结果，日志保存在“output”目录下。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集   | 精度（AP）  |  性能(aoe) |
| -------- | ----------- | -------- | --------- | --------- |
| 310P3    |    1        | coco2017 |   67.1%   | 185.28    |
