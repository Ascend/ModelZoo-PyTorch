# DeeplabV3模型-推理指导


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

DeeplabV3是一个经典的图像语义分割网络，在v1和v2版本基础上进行改进，多尺度(multiple scales)分割物体，设计了串行和并行的带孔卷积模块，采用多种不同的atrous rates来获取多尺度的内容信息，提出 Atrous Spatial Pyramid Pooling(ASPP)模块, 挖掘不同尺度的卷积特征，以及编码了全局内容信息的图像层特征，提升图像分割效果。
- 参考论文：[Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation." arXiv preprint arXiv:1706.05587 (2017).](https://arxiv.org/pdf/1706.05587.pdf)

- 参考实现：

   ```
   url=https://github.com/open-mmlab/mmsegmentation.git
   branch=master
   commit_id=fa1554f1aaea9a2c58249b06e1ea48420091464d
   model_name=DeeplabV3
   ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 1024 x 2048 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output  | FLOAT64  | batchsize x 1 x 1024 x 2048 | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动
  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

2. 获取源码。**<u>*此处获取指获取第三方开源代码仓的命令*</u>**

   ```
   git clone -b master https://github.com/open-mmlab/mmsegmentation.git 
   pip3 install mmcv-full==1.3.7
   cd mmsegmentation
   git reset --hard fa1554f1aaea9a2c58249b06e1ea48420091464d
   pip3 install -e . 
   cd ..
   ```

   目录结构如下：
   ```
   ├──mmsegmentation                         //开源仓目录
   ├──deeplabv3_torch_preprocess.py
   ├──deeplabv3_torch_postprocess.py
   ├──LICENCE
   ├──requirements.txt
   ├──README.md
   ├──modelzoo_level.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型将使用到[Cityscapes 验证集]()，请用户需自行获取数据集，上传数据集到服务器任意目录并解压,以"./datasets"。目录结构如下：

   ```
   ├──./datasets
         ├──cityscapes
            ├──leftImg8bit       //预处理需要的数据集      
            ├──gtFine
               ├──val           
                  ├──munster
                  ├──lindau
                  ├──frankfurt
                     ├──frankfurt_000001_083852_gtFine_polygons.json
                     ├──frankfurt_000001_083852_gtFine_labelTrainIds.png   //后处理时需要，请选对数据集
                     ├──frankfurt_000001_083852_gtFine_labelIds.png
                     ├──frankfurt_000001_083852_gtFine_instanceIds.png
                     ├──frankfurt_000001_083852_gtFine_color.png
               ├──train
               ├──test
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行deeplabv3_torch_preprocess.py脚本，完成预处理。

   ```
   python3 deeplabv3_torch_preprocess.py  ./datasets/cityscapes/leftImg8bit/val/ ./prep_dataset
   ```
   - 参数说明：
      - 第一个参数：数据集路径。
      - 第二个参数：处理完成的数据集路径。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      从源码包中获取[权重文件  deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/22.1.30/ATC%20DeeplabV3%28FP16%29%20from%20Pytorch%20-%20Ascend310.zip),可放置于任意路径下，以"./"为例。

   2. 导出onnx文件。

      1. 使用deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth权重文件导出onnx文件。

         运行mmsegmentation/tools/pytorch2onnx.py脚本。

         ```
         python3 mmsegmentation/tools/pytorch2onnx.py mmsegmentation/configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py --checkpoint ./deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth --output-file deeplabv3.onnx --shape 1024 2048
         ```

         获得deeplabv3.onnx文件。

      2. 优化ONNX文件。

         ```
         python3 -m onnxsim deeplabv3.onnx deeplabv3_sim_bs1.onnx --input-shape="1,3,1024,2048" --dynamic-input-shape
         ```

         获得deeplabv3_sim_bs1.onnx文件。

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

         ```
         atc --framework=5 --model=deeplabv3_sim_bs1.onnx --output=deeplabv3_bs1 --input_format=NCHW --input_shape="input:1,3,1024,2048" --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成deeplabv3_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```
      mkdir result
      python3 -m ais_bench  --model deeplabv3_bs1.om --input ./prep_dataset --output ./result --output_dirname=bs1 --outfmt BIN --batchsize=1  --device=0
      ```
      - 参数说明：
         - --model：模型类型。
         - --input：om模型推理输入文件路径。
         - --output：om模型推理输出文件路径。
         - --output_dirname：
         - --outfmt：输出格式
         - --batchsize：批大小。
         - --device：NPU设备编号。

        推理后的输出默认在当前目录./result/bs1下。


   3. 精度验证。

      调用脚本与数据集的gtFine/val比对，可以获得Accuracy数据，结果在终端和./result.txt中显示。

      ```
      python3 deeplabv3_torch_postprocess.py --output_path=./result/bs1 --gt_path=./datasets/cityscapes/gtFine/val --result_path=./result
      ```

      - 参数说明：
         - --output_path：ais_infer生成推理结果所在路径。
         - --gt_path：标签数据路径。
         - --result_path：为生成结果文件


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | :----------------: | ---------- | ---------- | --------------- |
|  Ascend310P  |  1   |   Cityscapes   |   79.12    |   4.910    |
