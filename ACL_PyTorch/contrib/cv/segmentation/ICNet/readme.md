# ICNet模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

ICNet主要研究具有挑战性的实时语义分割问题。它有许多实际应用，但在减少像素级标签推理的大量计算方面存在根本困难。为了应对这一挑战，我们提出了一种图像级联网络（ICNet），该网络在适当的标签指导下结合了多分辨率分支。我们对我们的框架进行了深入分析，并引入了级联特征融合单元来快速实现高质量的分割。我们的系统可以在单个GPU卡上进行实时推断，并在具有挑战性的数据集（如Cityscapes、CamVid和COCO Stuff）上评估出高质量的结果。



- 参考实现：

  ```
  url=https://github.com/liminn/ICNet-pytorch
  commit_id=da394fc44f4fbaff1b47ab83ce7121a96f375b03
  model_name=ICNet
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 1024 x 2048 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 19 x 1024 x 2048 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/liminn/ICNet-pytorch
   cd ICNet-pytorch
   git reset --hard da394fc44f4fbaff1b47ab83ce7121a96f375b03
   cd ..
   patch -p2 < ./ICNet.patch
   ```

2. 安装依赖。
   ```
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型支持cityscapes leftImg8bit的500张验证集。用户需要下载[leftImg8bit_trainvaltest.zip](http://www.cityscapes-dataset.com/downloads)和[gtFine_trainvaltest.zip](http://www.cityscapes-dataset.com/downloads)数据集，解压，将两个数据集放在./datasets/cityscapes/目录下。

    ```text
    .
    └──datasets
        └──cityscapes
            ├──gtFine
            |    ├──test
            |    ├──train
            |    └──val
            └──leftImg8bit
                ├──test
                ├──train
                └──val
    ```


2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行ICNet_preprocess脚本，完成预处理。

   ```
   python3 ICNet_preprocess.py ./datasets/cityscapes/ ./pre_dataset_bin 
   ```
   - 第一个参数为数据集地址
   - 第二个参数为预处理后数据保存地址



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从该目录下获取模型权重[rankid0_icnet_resnet50_192_0.687_best_model.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/ICNet/PTH/rankid0_icnet_resnet50_192_0.687_best_model.pth)

   2. 导出onnx文件。

      1. 使用ICNet_pth2onnx.py导出onnx文件。<u>***请用脚本名称替换xxx***</u>

         运行ICNet_pth2onnx.py脚本。

         ```
         python ICNet_pth2onnx.py rankid0_icnet_resnet50_192_0.687_best_model.pth ICNet.onnx
         ```

         获得ICNet.onnx文件。


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
         atc --framework=5 --model=ICNet.onnx --output=ICNet_bs${bs} --out_nodes="Resize_317:0" --input_format=NCHW --input_shape="actual_input_1: ${bs},3,1024,2048" --log=debug --soc_version=Ascend${chip_name} 
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***ICNet_bs${bs}.om***</u>模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
      python -m ais_bench --model=ICNet_bs${bs}.om --input=./pre_dataset_bin --output=./ --output_dirname=./result --batchsize=${batch_size}     
        ```

        -   参数说明：

             -   model：om模型地址
             -   input：预处理数据
             -   output：推理结果保存路径
             -   output_dirname:推理结果保存子目录

        推理后的输出保存在当前目录result下。


   3. 精度验证。

      调用ICNet_postprocess.py脚本进行精度计算

      ```
       python ICNet_postprocess.py ./datasets/cityscapes/ ./result/ ./out
      ```

      - 参数说明：

        - 第一个参数为数据集地址
        - 第二个参数为推理结果保存地址
        - 第三个结果为精度储存地址

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7 -m ais_bench --model=ICNet_bs${bs}.om --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型路径
        - --batchsize：batchsize大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|    Ascend310P3       |        1          |     cityscapes       |     mIoU: 0.689       |         19.25        |
|    Ascend310P3       |        4          |     cityscapes       |            |       31.22          |
|    Ascend310P3       |        8          |     cityscapes       |            |        32.33         |
|    Ascend310P3       |        16          |     cityscapes       |            |        31.24         |
|    Ascend310P3       |        32          |     cityscapes       |     内存不足，无法推理       |                 |