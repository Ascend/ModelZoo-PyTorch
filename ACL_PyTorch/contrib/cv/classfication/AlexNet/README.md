# AlexNet模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Alex在2012年提出的alexnet网络结构模型,首次在CNN中成功应用了ReLU、Dropout和LRN等Trick,引爆了神经网络的应用热潮,并赢得了2012届图像识别大赛的冠军,使得CNN成为在图像分类上的核心算法模型。


- 参考实现：

  ```
  url=https://github.com/pytorch/examples/tree/master/imagenet
  commit_id=49e1a8847c8c4d8d3c576479cb2fe2fd2ac583de
  code_path=https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/AlexNet
  model_name=AlexNet
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

  | 配套                                                         | 版本      | 环境准备指导                                                 |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.1   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   直接可以调用torch内的alexNet模型，无需下载源码
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）


   该模型使用[ImageNet官网](http://www.image-net.org/)的5万张验证集进行测试，图片与标签分别存放在/local/AlexNet/imagenet/val与/local/AlexNet/imagenet/val_label.txt。
   ```
   imagenet
   ├── val_label.txt    //验证集标注信息       
   └── val             // 验证集文件夹
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行imagenet_torch_preprocess.py脚本，完成预处理。

   ```
   python3.7 imagenet_torch_preprocess.py /local/AlexNet/imagenet/val ./pre_dataset

   ```
   
   - 参数说明：
   
     /local/AlexNet/imagenet/val，验证集文件所在路径
         
     ./pre_dataset，输出的预处理后数据集路径



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [AlexNet预训练pth权重文件](https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth)

      ```
      wget https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
      ```

   2. 导出onnx文件。

      1. 使用pth2onnx.py脚本。

         运行pth2onnx.py脚本。

         ```
         python3.7 pth2onnx.py alexnet-owt-4df8aa71.pth alexnet.onnx
         ```

         获得alexnet.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/......
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
         atc --model=./alexnet.onnx --framework=5 --output=./onnx_alexnet_bs{batch size} --input_format=NCHW --input_shape="actual_input_1:{batch size},3,224,224" --log=debug --soc_version=Ascend310P3
         示例
         atc --model=./alexnet.onnx --framework=5 --output=./onnx_alexnet_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,224,224" --log=debug --soc_version=Ascend310P3
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成onnx_alexnet_bs1.om模型文件，batch size为4、8、16、32、64的修改对应的batch size的位置即可。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3 -m ais_bench --model ./onnx_alexnet_bs{batch size}.om --input ./pre_dataset/ --output ./output --output_dirname subdir --outfmt 'TXT' --batchsize {batch size}
        示例
        python3 -m ais_bench --model ./onnx_alexnet_bs1.om --input ./pre_dataset/ --output ./output --output_dirname subdir --outfmt 'TXT' --batchsize 1
        ```

        -   参数说明：

             -   model：需要推理om模型的路径。
             -   input：模型需要的输入bin文件夹路径。
             -   output：推理结果输出路径。
             -   outfmt：输出数据的格式。
             -   output_dirname:推理结果输出子文件夹。

        推理后的输出默认在当前目录output的subdir下。

   3. 精度验证。

      调用vision_metric_ImageNet.py脚本与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。

      ```
       python3.7 vision_metric.py --benchmark_out ./output/subdir/ --anno_file /local/AlexNet/imagenet/val_label.txt --result_file ./result.json
      ```

      - 参数说明：

        - benchmark_out：为生成推理结果所在路径  

        - anno_file：为标签数据所在路径

        - result_file：为生成结果文件所在路径

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
        python3.7 -m ais_bench --model=./onnx_alexnet_bs{batch size}.om --loop=1000 --batchsize={batch size}
        示例
        python3.7 -m ais_bench --model=./onnx_alexnet_bs1.om --loop=1000 --batchsize=1
        ```

      - 参数说明：
        - --model：需要验证om模型所在路径
        - --batchsize：验证模型的batch size，按实际进行修改



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度                   | 性能    |
| --------- |------------| ---------- |----------------------|-------|
|   310P3        | 1          |  ImageNet          | 56.56/Top1 79.1/Top5 | 1266  |
|   310P3        | 4          |  ImageNet          | 56.56/Top1 79.1/Top5 | 4324  |
|   310P3        | 8          |  ImageNet          | 56.56/Top1 79.1/Top5 | 7266  |
|   310P3        | 16         |  ImageNet          | 56.56/Top1 79.1/Top5 | 9975  |
|   310P3        | 32         |  ImageNet          | 56.56/Top1 79.1/Top5 | 11055 |
|   310P3        | 64         |  ImageNet          | 56.56/Top1 79.1/Top5 | 12672 |