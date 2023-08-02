# MobileNetV2-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

mobileNetV2是对mobileNetV1的改进，是一种轻量级的神经网络。mobileNetV2保留了V1版本的深度可分离卷积，增加了线性瓶颈（Linear Bottleneck）和倒残差（Inverted Residual）。

- 参考实现：

  ```
  url=https://github.com/pytorch/vision
  commit_id=f15f4e83f06f7e969e4239c06dc17c7c9e7d731d
  model_name=MobileNetV2
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型    | 大小     | 数据排布格式 |
  |---------|--------|--------| ------------ |
  | input    | Float16 | batchsize x 3 x 224 x 224 | ND     |


- 输出数据

  | 输出数据 | 数据类型               | 大小    | 数据排布格式 |
  |------------------|---------| -------- | ------------ |
  | output  | FLOAT16 | batchsize x 1000 | ND           |



# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------------ |---------| ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   torch包可以获取模型结构，无需下载源码包

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在/usr/local/MobileNetV2_for_Pytorch/imagenet/val与/usr/local/MobileNetV2_for_Pytorch/imagenet/val_label.txt。
   
2. 数据预处理。

   执行预处理脚本，生成数据集预处理后的bin文件
    ```
    python3 preprocess.py /usr/local/MobileNetV2_for_Pytorch/imagenet/val/ ./preprocess_data
      
    运行成功后，当前目录下preprocess_data文件夹下生成bin格式的数据集
    ```

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       下载对应的[权重文件](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth)于MobileNetV2_for_Pytorch目录下。

       ```
       mkdir output
       ```

   2. 导出onnx文件。

      1. 使用**pth2onnx.py**导出onnx文件。

         运行**pth2onnx.py**脚本。

         ```
         python3 pth2onnx.py 
         ```
         > **说明：** 
         运行成功后在output文件夹下生成**mobilenetv2.onnx**模型文件。
      
   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

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
         bash atc.sh --model mobilenet_v2 --bs 4 --soc Ascend310P3
         ```

         - 参数说明：
         
           --model：ONNX模型文件
         
           --framework：5代表ONNX模型
         
           --output：输出的OM模型

           --input_format：输入数据的格式

           --input_shape：输入数据的shape

           --log：日志级别

           --soc_version：处理器型号

           运行成功后在output文件夹下生成**om**模型文件。

2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。


   b.  执行推理。

      ```
      python3 -m ais_bench --model ./output/mobilenet_v2_bs4.om --input ./preprocess_data --output ./output --output_dirname subdir --outfmt 'TXT' --batchsize 4
      ```
    
      -   参数说明：
    
           -   model：需要推理om模型的路径。
           -   input：模型需要的输入bin文件夹路径。
           -   output：推理结果输出路径。
           -   outfmt：输出数据的格式。
           -   output_dirname:推理结果输出子文件夹。
    	...


   c.  精度验证。

      调用脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。
    
      ```
      python3 postprocess.py  ./output/subdir  /usr/local/MobileNetV2_for_Pytorch/imagenet/val_label.txt ./ result.json
      ```
    
      ./output/subdir：为生成推理结果所在路径  
    
      /usr/local/MobileNetV2_for_Pytorch/imagenet/val_label.txt：为标签数据
    
      result.json：为生成结果文件,位于当前目录下


   d.  性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
    
      ```
      python3 -m ais_bench --model ./output/mobilenet_v2_bs${bs}.om --loop 1000 --batchsize ${bs}
    
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号  | Batch Size | 数据集      | 精度                    | 性能      |
|-------|------------|----------|-----------------------|---------|
| 310P3  | 4          | ImageNet | 71.87/Top1 90.32/Top5 | 7072fps |
| 310B1 | 4 | ImageNet | 71.87/Top1 90.32/Top5 | 1488.1fps |


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
