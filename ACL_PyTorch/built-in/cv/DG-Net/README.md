# DG-net模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

DG-Net有机地将GAN和re-id backbone结合来解决行人重识别问题（person re-id）GAN和re-id backbone具有相同的appearance encoder，这样一来，利用GAN进行数据增强和重识别便不在是两个分割的过程，而是可以对两个过程同时进行训练优化，GAN存在的目的不仅仅是生成更多的图片，而是为解决re-id问题生成更多的图片。



- 参考实现：

  ```
  url=https://github.com/NVlabs/DG-Net
  commit_id=c7771598648520d960362912c6217a298a5d6ab8
  model_name=DG-net
  ```
  




## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input1    | RGB_FP32 | batchsize x 1 x 256 x 128 | NCHW         |
  | input2    | RGB_FP32 | batchsize x 3 x 256 x 128 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 3 x 256 x 128 | NCHW   




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.1.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/NVlabs/DG-Net.git
   cd DG-Net
   git reset --hard c7771598648520d960362912c6217a298a5d6ab8
   cd ..
   patch -p2 < DG-net.patch

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
   利用源码中的prepare-market.py脚本，将数据分类
   ```
   cd DG-net
   python prepare-market.py
   cd ..
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行DGnet_preprocess脚本，完成预处理。(在进行预处理之前需要先下载并解压权重文件)

   ```
   python3 ./DGnet_preprocess.py --input_folder=./Market/pytorch/train_all/ --output_folder=./bin_path1 --output_folder2=./bin_path2 
   ```
   - 参数说明：

      -   --input_folder：数据集
      -   --output_folder：输出第一个预处理数据
      -   --output_folder2：输出第二个预处理数据



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      在[目录](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/gan/DG-Net/DG-Net.zip)下自行获取权重文件，并解压

   2. 导出onnx文件。

      1. 使用DGnet_pth2onnx.py导出onnx文件。

         运行DGnet_pth2onnx.py脚本。

         ```
         python3 DGnet_pth2onnx.py --output=DG-net_bs${bs}.onnx --batchsize=${bs}
         ```

         获得DG-net_bs${bs}.onnx文件。


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
         atc --framework=5 --input_shape="image1:${bs},1,256,128;image2:${bs},3,256,128" --output=DG-net_bs${bs} --soc_version=Ascend310P3 --keep_dtype=keep.cfg --model=DG-net_bs${bs}.onnx 
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***DG-net_bs${bs}.om***</u>模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
      python3.7 -m ais_bench --model=DG-net_bs${bs}.om --input="./bin_path2,./bin_path1" --batchsize=${batch_size} --output=./results
        ```

        -   参数说明：

             -   model：om模型路径
             -   input：预处理数据
             -   batchsize：batchsize大小


   3. 精度验证。

      调用脚本对推理数据进行可视化处理

      ```
      python3 DGnet_postprocess.py --result_folder=./results/{time_line} --output_folder=./off-gan_id/ --output_folder2=./off-gan_bg/
      ```

      - 参数说明：

        - result_folder：推理结果保存路径


        - output_folder：可视化数据保存路径1


        - output_folder2：可视化数据保存路径2

      利用[TTUR](https://github.com/layumi/TTUR)源码仓工具计算生成数据集的FID值
      ```
      git clone https://github.com/layumi/TTUR.git
      cd TTUR
      python3 fid.py ../Market/pytorch/train_all ../off-gan_id
      ```

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7 -m ais_bench --model=DG-net_bs${bs}.om --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型地址
        - --batchsize：batchsize大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|     Ascend310P3      |        1          |      Market1501      |     18.12       |     296            |
|     Ascend310P3      |        4          |      Market1501      |            |       568          |
|     Ascend310P3      |        8          |      Market1501      |            |        584         |
|     Ascend310P3      |        16          |      Market1501      |            |        517         |
|     Ascend310P3      |        32          |      Market1501      |            |         467        |
|     Ascend310P3      |        64          |      Market1501      |            |         324        |