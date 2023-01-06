# DnCNN模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

DnCNN作为去噪神经网络非常出名，这个网络强调了residual learning（残差学习）和batch normalization（批量标准化）在信号复原中相辅相成的作用，可以在较深的网络的条件下，依然能带来快的收敛和好的性能。这个算法可以解决处理未知噪声水平的高斯去噪、超分辨率、JPEG去锁等多个领域的问题。


- 参考实现：

  ```
  url=https://github.com/SaoYan/DnCNN-PyTorch
  commit_id=6b0804951484eadb7f1ea24e8e5c9ede9bea485b
  code_path=https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/ACL_PyTorch/contrib/cv/image_process/DnCNN
  model_name=DnCNN
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | ---- |---------------------------| ------------------------- | ------------ |
  | input    | FP32 | batchsize x 1 x 481 x 481 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 1 x 481 x 481 | NCHW           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本      | 环境准备指导                                                 |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/SaoYan/DnCNN-PyTorch
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）


   该模型使用官网提供的数据集进行验证，存放路径为源码路径下的的data目录。

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行data_preprocess.py脚本，完成预处理。

   ```
   python3.7 data_preprocess.py ./DnCNN-PyTorch/data ISource INoisy

   ```
   
   - 参数说明：
   
     ./DnCNN-PyTorch/data，验证集文件所在路径
   
     ISource，输出的预处理后标签数据集路径     

     INoisy，输出的预处理后数据集路径



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [DnCNN预训练pth权重文件](https://www.hiascend.com/zh/software/modelzoo/models/detail/1/4ab8abf42ef54bb9b461aca384c6313e/1)

      ```
      进入网页点击立即下载，压缩包中有net.pth的权重文件
      ```

   2. 导出onnx文件。

      1. 使用DnCNN_pth2onnx.py脚本。

         运行DnCNN_pth2onnx.py脚本。

         ```
         python3.7 DnCNN_pth2onnx.py net.pth DnCNN-S-15.onnx
         ```

         获得DnCNN-S-15.onnx文件。

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
         atc --framework=5 --model=./DnCNN-S-15.onnx --input_format=NCHW --input_shape="actual_input_1:{batch size},1,481,481" --output=DnCNN-S-15_bs{batch size} --log=debug --soc_version=Ascend310P3
         示例
         atc --framework=5 --model=./DnCNN-S-15.onnx --input_format=NCHW --input_shape="actual_input_1:1,1,481,481" --output=DnCNN-S-15_bs1 --log=debug --soc_version=Ascend310P3
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
        python3 -m ais_bench --model ./DnCNN-S-15_bs{batch size}.om --input ./INoisy/ --output ./output --output_dirname subdir --outfmt 'BIN' --batchsize {batch size}
        示例
        python3 -m ais_bench --model ./DnCNN-S-15_bs1.om --input ./INoisy/ --output ./output --output_dirname subdir --outfmt 'BIN' --batchsize 1
        ```

        -   参数说明：

             -   model：需要推理om模型的路径。
             -   input：模型需要的输入bin文件夹路径。
             -   output：推理结果输出路径。
             -   outfmt：输出数据的格式。
             -   output_dirname:推理结果输出子文件夹。

        推理后的输出默认在当前目录output的subdir下。

   3. 精度验证。

      调用postprocess.py脚本推理结果进行PSRN计算，结果会打印在屏幕上。

      ```
       python3.7 postprocess.py ./output/subdir
      ```

      - 参数说明：

        - ./output/subdir：为生成推理结果所在路径

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
        python3.7 -m ais_bench --model=./DnCNN-S-15_bs{batch size}.om --loop=1000 --batchsize={batch size}
        示例
        python3.7 -m ais_bench --model=./DnCNN-S-15_bs1.om --loop=1000 --batchsize=1
        ```

      - 参数说明：
        - --model：需要验证om模型所在路径
        - --batchsize：验证模型的batch size，按实际进行修改



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集      | 精度                   | 性能  |
| --------- |------------|----------|----------------------|-----|
|   310P3        | 1          | 官网提供     | 31.53  | 128 |
|   310P3        | 4          | 官网提供 | 31.53 | 138 |
|   310P3        | 8          | 官网提供 | 31.53 | 153 |
|   310P3        | 16         | 官网提供 | 31.53 | 166 |
|   310P3        | 32         | 官网提供 | 31.53 | 142 |
|   310P3        | 64         | 官网提供 | 31.53 | 142 |