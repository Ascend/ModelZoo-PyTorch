# Swin-Transformer_tiny模型-推理指导

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

Swin-Transformer是针对于图片处理设计的基于Transformer架构的神经网络。该网络针对原始Transformer迁移到图片端后计算量过大，复用困难的问题，提出了新的swin-block以代替原有的attention架构。模型以窗口的attention方式极大地减少了图像不同区域间的互相响应，同时也避免了部分冗余信息的产生。最终，模型在减少了大量计算量的同时，在不同的视觉传统任务上也有了效果的提升。


- 参考实现：

  ```
  url=https://github.com/microsoft/Swin-Transformer
  branch=master
  commit_id=014eb33148a5e41576dd91715d5c557896613f51
  ```
## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | image    | FLOAT32  | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 大小              | 数据类型 | 数据排布格式 |
  | -------- | --------          | -------- | ------------ |
  | class    | batch_size x 1000 | FLOAT32  | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                            | 版本    | 环境准备指导                                                                                          |
| ------------------------------------------------------------    | ------- | ------------------------------------------------------------                                          |
| 固件与驱动                                                      | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                            | 6.0.RC1 | -                                                                                                     |
| Python                                                          | 3.7.5   | -                                                                                                     |
| PyTorch                                                         | 1.8.0+  | -                                                                                                     |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master         # 切换到对应分支
   cd ACL_PyTorch/contrib/cv/Swin-Transformer_tiny              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

2. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/microsoft/Swin-Transformer
   cd Swin-Transformer
   git checkout 6bbd83ca617db8480b2fb9b335c476ffaf5afb1a
   patch -p1 < ../swin.patch
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型使用val_label.txt文件作为最后的比对标签。

   数据目录结构请参考：
   ```
   ├──ImageNet
    ├──ILSVRC2012_img_val
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行“swin_preprocess.py”脚本，完成预处理。

   ```
   mkdir prep_data
   python3 swin_preprocess.py --cfg=Swin-Transformer/configs/swin_tiny_patch4_window7_224.yaml --data-path=${datasets_path}/imagenet --bin_path=prep_data
   ```
   --cfg：模型配置文件。

   --data-path：原始数据验证集（.jpeg）所在路径。

   --bin_path：输出的二进制文件（.bin）所在路径。

   每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成二进制文件夹。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       获取权重文件“swin_tiny_patch4_window7_224.pth: [[权重文件下载链接](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)]

   2. 导出onnx文件。

      1. 使用脚本导出onnx文件。

         运行“swin_pth2onnx.py”脚本。

         ```
         # 以bs1为例
         mkdir -p models/onnx
         python3 swin_pth2onnx.py --resume=swin_tiny_patch4_window7_224.pth --cfg=Swin-Transformer/configs/swin_tiny_patch4_window7_224.yaml --batch-size=16
         ```

         获得models/onnx/swin_tiny_bs16.onnx文件。

      2. 优化ONNX文件。

         ```
         python3 -m onnxsim models/onnx/swin_tiny_bs16.onnx models/onnx/swin_tiny_bs16_sim.onnx
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
         mkdir -p models/om
         atc --framework=5 --model=models/onnx/swin_tiny_bs16_sim.onnx  --output=models/om/swin_tiny_bs16 --input_format=NCHW --input_shape="image:16,3,224,224" --log=debug --soc_version=${chip_name} --enable_small_channel=1 --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成swin_b16.om模型文件。



2. 开始推理验证。

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
        # 以bs16为例
        mkdir -p outputs/bs16
        python3 -m ais_bench --model models/om/swin_tiny_bs16.om --input prep_data --output outputs --output_dirname bs16 --device 1
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --device：NPU设备编号。


        推理后的输出默认在当前目录outputs/bs1下。

        >**说明：**
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   3.  精度验证。

      调用swin_postprocess.py脚本与数据集标签target.json比对，可以获得Accuracy数据，结果保存在result.json中。

  

      ```
      python3 swin_postprocess.py --input_dir=outputs/bs16/ --label_path=val_label.txt --save_path=./result_bs16.json
      ```
      注：--input_dir指定的路径不是固定，具体路径为ais-infer工具推理命令中'--output/--output_dirname'指定目录下的生成推理结果所在路径

      --input_dir：生成推理结果所在路径

      --label_path：标签数据文件路径

      --save_path：生成结果文件

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度参考下列数据:

| device |   top1 |  top5 |
|--------|--------|-------|
| 基准   |      - |     - |
| 310    | 81.19% | 95.5% |
| 310P   | 81.15% | 95.42% |

性能参考下列数据。

| batchsize | 1        | 4        | 8        | 16       | 32       | 64        |
|-----------|----------|----------|----------|----------|----------|-----------|
| 310       | 201.9fps | 253.3fps | 243.4fps | 237.6fps | 238.8fps | 243.2fps  |
| 310P      | 339.7fps | 507.2fps | 564.7fps | 482.3fps | 434.2fps | 406.9fps  |
| 基准      | 267.7fps | 384.7fps | 387.6fps | 270.8fps | 232.6fps | 188.97fps |
