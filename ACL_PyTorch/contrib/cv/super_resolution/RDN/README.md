# RDN模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

RDN用于图像超分辨率的残差密集网络,主要是提出了网络结构RDB(residual dense blocks)，它本质上就是残差网络结构与密集网络结构的结合。
- 参考实现：

  ```
  url=https://github.com/yjn870/RDN-pytorch
  commit_id=f2641c0817f9e1f72acd961e3ebf42c89778a054
  code_path=https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/super_resolution/RDN
  model_name=RDN
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- |---------------------------| ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 114 x 114 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- |---------------------------| -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 3 x 228 x 228 | NCHW           |



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
   git clone https://github.com/yjn870/RDN-pytorch.git
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）


   RDN模型使用[Set5验证集](https://github.com/hengchuan/RDN-TensorFlow/tree/master/Test/Set5)的5张图片进行测试，图片存放在/local/RDN/dataset/set5下面。
   

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行RDN_preprocess.py脚本，完成预处理。

   ```
   python3.7 RDN_preprocess.py --src-path=/local/RDN/dataset/set5 --save-path=./prep_dataset

   ```
   
   - 参数说明：
   
     src-path，原始数据验证集（.jpeg）所在路径。
         
     save-path，输出的二进制文件（.bin）所在路径。



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       [RDN_x2预训练pth权重文件](https://www.dropbox.com/s/pd52pkmaik1ri0h/rdn_x2.pth?dl=0)
      
   2. 导出onnx文件。

      1. 使用RDN_pth2onnx.py脚本。

         运行RDN_pth2onnx.py脚本。

         ```
         python3.7 RDN_pth2onnx.py --input-file=rdn_x2.pth --output-file=rdn_x2.onnx
         ```

         获得rdn_x2.onnx文件。

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
         atc --framework=5 --model=rdn_x2.onnx --output=rdn_x2_bs{batch size} --input_format=NCHW --input_shape="image:{batch size},3,114,114" --log=debug --soc_version=Ascend310P3
         示例
         atc --framework=5 --model=rdn_x2.onnx --output=rdn_x2_bs1 --input_format=NCHW --input_shape="image:1,3,114,114" --log=debug --soc_version=Ascend310P3
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成rdn_x2_bs1.om模型文件，batch size为4、8、16、32、64的修改对应的batch size的位置即可。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3 -m ais_bench --model ./rdn_x2_bs{batch size}.om --input ./prep_dataset/data --output ./output --output_dirname subdir --outfmt 'TXT' --batchsize {batch size}
        示例
        python3 -m ais_bench --model ./rdn_x2_bs1.om --input ./prep_dataset/data --output ./output --output_dirname subdir --outfmt 'TXT' --batchsize 1
        ```

        -   参数说明：

             -   model：需要推理om模型的路径。
             -   input：模型需要的输入bin文件夹路径。
             -   output：推理结果输出路径。
             -   outfmt：输出数据的格式。
             -   output_dirname:推理结果输出子文件夹。

        推理后的输出默认在当前目录output的subdir下。

   3. 精度验证。

      调用RDN_postprocess.py脚本推理结果，结果保存在result.json中。

      ```
      python3.7 RDN_postprocess.py --pred-path=./output/subdir --label-path=./prep_dataset/label --result-path=./result.json --width=114 --height=114 --scale=2
      ```

      - 参数说明：

        - pred-path：为生成推理结果所在路径  

        - label-path：为标签数据所在路径

        - result-path：结果保存路径

   4. 性能验证。

     可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

       ```
       python3.7 -m ais_bench --model=./rdn_x2_bs{batch size}.om --loop=1000 --batchsize={batch size}
       示例
       python3.7 -m ais_bench --model=./rdn_x2_bs1.om --loop=1000 --batchsize=1
       ```

     - 参数说明：
       - --model：需要验证om模型所在路径
       - --batchsize：验证模型的batch size，按实际进行修改



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度                 | 性能  |
| --------- |------------| ---------- |--------------------|-----|
|   310P3        | 1          |  Set5          | 38.27| 47  |
|   310P3        | 4          |  Set5         | 38.27 | 37  |
|   310P3        | 8          |  Set5         | 38.27 | 33  |
|   310P3        | 16         |  Set5         | 38.27 | 33  |
|   310P3        | 32         |  Set5         | 38.27 | 33  |
|   310P3        | 64         |  Set5         | 38.27 | 33  |