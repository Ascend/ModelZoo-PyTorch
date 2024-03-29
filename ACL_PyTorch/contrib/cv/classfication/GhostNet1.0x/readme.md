# GhostNet1.0x模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

 GhostNet是华为诺亚方舟实验室提出的一个新型神经网络结构，其中的Ghost Module和深度分离卷积就很类似，不同之处在于先进行PointwiseConv，后进行DepthwiseConv，另外增加了DepthwiseConv的数量，包括一个恒定映射。

- 参考实现：

  ```
  url=https://github.com/huawei-noah/CV-Backbones
  branch=master
  commit_id=5a06c87a8c659feb2d18d3d4179f344b9defaceb
  model_name=GhostNet1.0x
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 1000 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/huawei-noah/CV-Backbones.git
   cd CV-Backbones
   git reset --hard 5a06c87a8c659feb2d18d3d4179f344b9defaceb
   cd ..
   ```

2. 安装依赖。

   ```
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   本模型使用[ImageNet官网](https://gitee.com/link?target=http%3A%2F%2Fwww.image-net.org)的5万张验证集进行测试，以ILSVRC2012为例，用户需获取[ILSVRC2012数据集](http://www.image-net.org/download-images)，并上传到服务器，图片与标签分别存放在./imagenet/val与./imageNet/val_label.txt。
   ```
   ├── imagenet
       ├── val
       ├── val_label.txt 
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行Ghostnet_preprocess.py脚本，完成预处理。

   ```
   python Ghostnet_preprocess.py ./imagenet/val ./prep_dataset
   ```
   - 参数说明：

      -   第一个参数为数据集目录。
      -   第二个参数为预处理保存目录。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      下载pth权重文件[GhostNet预训练pth权重文件](https://github.com/huawei-noah/CV-Backbones/raw/master/ghostnet_pytorch/models/state_dict_73.98.pth)或是执行下述命令获取
       ```
       wget http://github.com/huawei-noah/CV-Backbones/raw/master/ghostnet_pytorch/models/state_dict_73.98.pth
       ```

   2. 导出onnx文件。

      1. 使用Ghostnet_pth2onnx.py导出onnx文件。

         运行Ghostnet_pth2onnx.py脚本。

         ```
         python Ghostnet_pth2onnx.py state_dict_73.98.pth Ghostnet.onnx
         ```

         获得Ghostnet.onnx文件。

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
         atc --framework=5 \
             --model=./Ghostnet.onnx \
             --input_format=NCHW \
             --input_shape="image:${bs},3,224,224" \
             --output=ghostnet_bs${bs} \
             --log=debug \
             --soc_version=Ascend${chip_name}          
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***ghostnet_bs${bs}.om***</u>模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
      python -m ais_bench \
             --model=ghostnet_bs${bs}.om \
             --input=./prep_dataset \
             --output=./ \
             --output_dirname=./result \
             --batchsize=${batch_size}     
        ```

        -   参数说明：

             -   model：om模型地址
             -   input：预处理数据
             -   output：推理结果保存路径
             -   output_dirname:推理结果保存子目录

        推理后的输出默认在当前目录result下。


   3. 精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据。

      ```
      python Ghostnet_postprocess.py --result_path=result_summary.json --gt_path=./imagenet/val_label.txt
      ```

      - 参数说明：

        - result_path：推理结果生成的json信息文件


        - gt_path：为标签数据

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
      python -m ais_bench --model=ghostnet_bs${bs}.om --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型路径
        - --batchsize：batchsize大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|    Ascend310P3       |         1         |     imagenet       |     73.98%       |      1801           |
|    Ascend310P3       |         4         |     imagenet       |            |       3632          |
|    Ascend310P3       |         8         |     imagenet       |            |       4974          |
|    Ascend310P3       |         16         |     imagenet       |            |       4292          |
|    Ascend310P3       |         32         |     imagenet       |            |       3519          |
|    Ascend310P3       |         64         |     imagenet       |            |       3058          |