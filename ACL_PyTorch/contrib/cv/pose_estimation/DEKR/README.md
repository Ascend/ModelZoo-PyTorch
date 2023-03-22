# DEKR模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

DEKR采用的是自底向上的范式，准确地回归关键点位置需要学习专注于关键点区域的表征，通过采用自适应卷积来激活关键点区域中的像素。模型使用一个多分支结构进行独立的回归：每个分支学习一个具有专用自适应卷积的表示，并回归一个关键点。由此得到的解构式表示能够分别关注关键点区域，因此关键点回归在空间上更准确。



- 参考实现：

  ```
  url=https://github.com/HRNet/DEKR.git
  branch=master
  commit_id=7a303139e92bdf3eab8d899415ccac37374285a4
  model_name=DEKR(pose_hrnet_w32)
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                                                         | 数据排布格式 |
  | -------- | -------- | ------------------------------------------------------------ | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 512 x 768;<br/>batchsize x 3 x 512 x 512;<br/>batchsize x 3 x 768 x 512;<br/>batchsize x 3 x 512 x 1024;<br/>batchsize x 3 x 1024 x 512 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                                                         | 数据排布格式 |
  | -------- | ------------ | -------------------------------------------------------- | ------------ |
  | output1  | FLOAT32  | batchsize x 18 x 128 x 192;<br/>batchsize x 18 x 128 x 128;<br/>batchsize x 18 x 192 x 128;<br/>batchsize x 18 x 128 x 256;<br/>batchsize x 18 x 256 x 128 | NCHW         |
  | output2  | FLOAT32  | batchsize x 34 x 128 x 192;<br/>batchsize x 34 x 128 x 128;<br/>batchsize x 34 x 192 x 128;<br/>batchsize x 34 x 128 x 256;<br/>batchsize x 34 x 256 x 128 | NCHW         |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                    | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>
## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```shell
   git clone https://github.com/HRNet/DEKR.git
   cd DEKR
   git reset --hard 7a303139e92bdf3eab8d899415ccac37374285a4
   patch -p1 < ../DEKR.patch
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

2. 获取源代码



## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   ```
   mkdir data
   ```
   推理数据集采用 COCO_Val 2017，将 person_keypoints_val2017.json 文件和 val2017.zip 文件上传到主目录data文件夹下，目录结构如下：

   ```
   data
   |-- coco
   `-- |-- annotations
       |   `-- person_keypoints_val2017.json
       `-- images
           `-- val2017.zip
   ```


2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   在代码目录执行 DEKR_preprocess.py 脚本，完成预处理。

   ```
   python3 DEKR_preprocess.py  --output ./prep_data  --output_flip ./prep_data_flip  DATASET.ROOT data/coco
   ```

   - 参数说明：

     -  DATASET.ROOT：原始数据验证集所在路径。
     -  --output：原始图像输出的二进制文件（.npy）所在路径。
     -  --output_flip：原始图像flip后输出的二进制文件（.npy）所在路径。

   - 注：每个图像对应生成两个二进制文件。

   运行成功后，分别在 prep_data 和 prep_data_flip 两个文件夹下生成对应的 npy 文件。




## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用 PyTorch 将模型权重文件 .pth 转换为 .onnx 文件，再使用 ATC 工具将 .onnx 文件转为离线推理模型文件 .om 文件。

   1. 获取权重文件。

      - [权重链接](https://mailustceducn-my.sharepoint.com/:f:/g/personal/aa397601_mail_ustc_edu_cn/EmoNwNpq4L1FgUsC9KbWezABSotd3BGOlcWCdkBi91l50g?e=HWuluh)
      - 找到 model/pose_coco/pose_dekr_hrnetw32_coco.pth 和 model/rescore/final_rescore_coco_kpt.pth 两个权重文件，下载并放在 models 文件夹下。
      ```
      mkdir -p models
      mv pose_dekr_hrnetw32_coco.pth models
      mv final_rescore_coco_kpt.pth models
      ```


   2. 导出 onnx 文件。

      1. 使用 DEKR_pth2onnx.py 导出onnx文件。

         运行 DEKR_pth2onnx.py 脚本。

         ```
         python3 DEKR_pth2onnx.py --output models/dekr_bs1.onnx TEST.MODEL_FILE models/pose_dekr_hrnetw32_coco.pth
         ```

         获得 dekr_bs1.onnx 文件。
         
         - 参数说明：
         
           -  --output: onnx文件的输出路径。
           -  TEST.MODEL_FILE: pth权重文件所在路径。

   3. 使用 ATC 工具将 ONNX 模型转 OM 模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（chip_name）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3
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

      3. 执行 ATC 命令。

         ```shell
         atc --framework=5 \
             --model=models/dekr_bs1.onnx \
             --output=models/dekr_bs1 --input_format=ND \
             --input_shape="image:1,3,-1,-1" \
             --dynamic_dims="512,768;512,512;768,512;512,1024;1024,512" \
             --soc_version=Ascend${chip_name} \
             --log=error
         ```
         
         - 参数说明：
         
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --dynamic_dims：图片的动态分辨率参数。
           -   --log：设置ATC模型转换过程中显示日志的级别。
           -   --soc\_version：处理器型号。
           
           运行成功后生成 dekr_bs1.om 模型文件。



2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

      ```shell
      python3 -m ais_bench \
            --model=models/dekr_bs1.om \
            --input=./prep_data \
            --output=./ --output_dirname=out \
            --outfmt NPY \
            --auto_set_dymdims_mode 1
      python3 -m ais_bench \
            --model=higher_hrnet_dynamic.om \
            --input=./prep_data_flip \
            --output=./ --output_dirname=out_flip \
            --outfmt NPY \
            --auto_set_dymdims_mode 1
      ```
   
      - 参数说明：

         - model：om模型。
         - input：模型需要的输入。
         - output：推理结果输出路径。
         - outfmt：输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”。
         - output_dirname:推理结果输出子文件夹。可选参数。与参数output搭配使用。
         - auto_set_dymdims_mode：自动匹配输入数据的shape。

   3. 精度验证。

      运行 DEKR_postprocess.py 脚本，与 final_rescore_coco_kpt.pth 比对，可以获得精度数据。
      ```shell
      python3 DEKR_postprocess.py \
         --dump_dir './out' \
         --dump_dir_flip './out_flip' \
         RESCORE.MODEL_FILE models/final_rescore_coco_kpt.pth
      ```

      - 参数说明：
         -  --dump_dir: 原始图像执行推理后的输出，默认为'./out'。
         -  --dump_dir_flip: 原始图像flip后执行推理的输出，默认为'./out_flip'。
         -  RESCORE.MODEL_FILE: 验证模型文件所在路径。
   



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用 ACL 接口推理计算，性能参考下列数据。

| 芯片型号  | Batch Size |    数据集     | 精度      | 性能       |
| ------- | ---------- | ------------- | --------- | ---------- |
| 310P3   |     1      |   coco2017    | AP: 0.677 |  7.72      |

- 说明：该模型只支持bs1。