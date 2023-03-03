# OSNet模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

作为一个实例级的识别问题，行人再识别(ReID)依赖于具有识别能力的特征，它不仅能捕获不同的空间尺度，还能封装多个尺度的任意组合。这些同构和异构尺度的特征为全尺度特征。本文设计了一种新颖的深度CNN，称为全尺度网络(OSNet)，用于ReID的全尺度特征学习。这是通过设计一个由多个卷积特征流组成的残差块来实现的，每个残差块检测一定尺度的特征。重要的是，引入了一种新的统一聚合门用输入依赖的每个通道权重进行动态多尺度特征融合。为了有效地学习空间通道相关性，避免过拟合，构建块同时使用点卷积和深度卷积。通过逐层叠加这些块，OSNet非常轻量，可以在现有的ReID基准上从零开始训练。尽管OSNet模型很小，但其在6个Reid数据集上到达了SOTA结果。


- 参考实现：

  ```
  url=https://github.com/KaiyangZhou/deep-person-reid
  commit_id=e580b699c34b6f753a9a06223d840317546c98aa
  model_name=OSNet
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 256 x 128 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 512 | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   <u>***请核对CANN版本与torch版本是否符合当前模型要求，并按对应CANN版本填写固件版本，查询链接：https://www.hiascend.com/hardware/firmware-drivers?tag=commercial***</u>

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.11.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone  https://github.com/KaiyangZhou/deep-person-reid     
   cd deep-person-reid
   git reset --hard e580b699c34b6f753a9a06223d840317546c98aa
   python setup.py develop
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持Market-1501数据集，基本结构目录如下：

   ```
   Market
   ├── gt_query
   ├── bounding_box_test    
   ├── query
   ├── gt_bbox 
   └── bounding_box_train
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行OSNet_preprocess.py脚本，完成预处理。

   ```
   #处理gallery数据集，即bounding_box_test测试集
   mkdir gallery_prep_dataset
   python OSNet_preprocess.py Market/bounding_box_test ./gallery_prep_dataset/
   # 处理query数据集
   mkdir query_prep_dataset
   python OSNet_preprocess.py Market/query ./query_prep_dataset/ 
   ```




## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       在该目录下获取权重文件[osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/OSNet/PTH/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth)

   2. 导出onnx文件。

      1. 使用OSNet_pth2onnx.py导出onnx文件。
         运行OSNet_pth2onnx.py脚本。

         ```
         python OSNet_pth2onnx.py osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth osnet_x1_0.onnx
         ```

         获得 osnet_x1_0.onnx文件。

      2. 优化ONNX文件。

         ```
         python -m onnxsim osnet_x1_0.onnx osnet_x1_0_bs${bs}_sim.onnx --input-shape ${bs},3,256,128
         ```

         获得osnet_x1_0_bs${bs}_sim.onnx文件。

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
         atc --framework=5 --model=./osnet_x1_0_bs${bs}_sim.onnx --input_format=NCHW --input_shape="image:${bs},3,256,128" --output=osnet_x1_0_bs${bs} --soc_version=Ascend${chip_name} 
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***osnet_x1_0_bs${bs}.om***</u>模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
      python -m ais_bench --model=osnet_x1_0_bs${bs}.om --input=./gallery_prep_dataset/ --output=./ --output_dirname=./result_gallery --batchsize=${batch_size} --outfmt=TXT  

      python -m ais_bench --model=osnet_x1_0_bs${bs}.om --input=./query_prep_dataset/ --output=./ --output_dirname=./result_query --batchsize=${batch_size}  --outfmt=TXT
        ```

        -   参数说明：

             -   model：om模型地址
             -   input：预处理数据
             -   output：推理结果保存路径
             -   output_dirname：推理结果保存子目录
             -   outfmt：输出数据格式

        推理后的输出保存在当前目录result_query和result_gallery下。


   3. 精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

      ```
      python OSNet_postprocess.py  result_query result_gallery ./ result.json
      ```

      - 参数说明：

        - 第一个参数为query数据的推理结果


        - 第二个参数为gallery数据的推理结果


        - 第三个参数为精度结果的保存路径

        - 第四个参数为精度结果的保存json文件

   4. 性能验证。<u>***补充参数说明***</u>

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7 -m ais_bench --model=osnet_x1_0_bs${bs}.om --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型路径
        - --batchsize：batchsize大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|     Ascend310P3      |       1           |     Market-1501       |    R1:94.38% mAP:82.55%        |       1314          |
|     Ascend310P3      |       4           |     Market-1501       |            |        3529         |
|     Ascend310P3      |       8           |     Market-1501       |            |       4075         |
|     Ascend310P3      |       16           |     Market-1501       |            |        3649         |
|     Ascend310P3      |       32           |     Market-1501       |            |       3321          |
|     Ascend310P3      |       64           |     Market-1501       |            |       2821          |