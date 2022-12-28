# PraNet模型-推理指导


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

该网络主要用于分割结肠镜图像中的息肉，而同一类型的息肉具有大小、颜色和纹理的多样性;且息肉与周围粘膜的边界不清晰。为了解决这些挑战，该网络先并行部分解码器(PPD)聚合高级层中的特征，然后，根据组合的特征，生成一个全局地图，作为以下组件的初始指导区域。此外，利用反向注意(RA)模块挖掘边界线索，该模块能够建立区域与边界线索之间的关系。该策略有三个优势，即学习能力更好，泛化能力更好，训练效率更高。


- 参考实现：

  ```
  url=https://github.com/DengPingFan/PraNet.git
  commit_id=f697d5f566a4479f2728ab138401b7476f2f65b9
  model_name=contrib/cv/segmentation/PraNet
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 352 x 352 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 1 x 352 x 352 | NCHW         |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/DengPingFan/PraNet.git -b master
   cd PraNet
   git reset --hard f697d5f566a4479f2728ab138401b7476f2f65b9
   patch -p1 < ../PraNet_perf.diff
   cd ..
   ```
   > **说明**：因开源代码仓使用matlab评测，故需从[https://github.com/plemeri/UACANet](https://github.com/plemeri/UACANet)获取pytorch实现的评测脚本eval_functions.py（在utils目录下），并将其放在utils目录下

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持[kvasir](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view)，解压后目录结构如下：

   ```
   kvasir
   ├── images      
   └── masks
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行`PraNet_preprocess.py`脚本，完成预处理。
   ```
   python3 PraNet_preprocess.py ./Kvasir ./prep_bin 
   ```



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       ```
       wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Pranet/PTH/PraNet-19.pth
       ```

   2. 导出onnx文件。

      1. 使用pth文件导出onnx文件，运行PraNet_pth2onnx.py脚本。

         运行PraNet_pth2onnx.py脚本。

         ```
         python3.7 PraNet_pth2onnx.py   ./PraNet-19.pth  ./PraNet-19.onnx
         ```

         获得`PraNet-19.onnx`文件。



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
          atc --framework=5 --model=PraNet-19bs1.onnx --output=PraNet-19_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,352,352"  --log=error  --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
          

           运行成功后生成`PraNet-19_bs1.om`模型文件。

2. 开始推理验证。

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
         python3 -m ais_bench --model PraNet-19_bs1.om --input ./prep_bin --output ./ --output_dirname result
        ```

        -   参数说明：

             -   model：om模型 
             -   input：输入数据
             -   output：输出结果路径
             -   output_dirname: 输出结果文件夹
                  	
        推理后的输出默认在当前目录`result`下。

        >**说明：** 
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   3. 精度验证。

      调用`PraNet_postprocess.py`和`Eval.py`脚本，可以获得结果

      ```
       python3.7 PraNet_postprocess.py  ./Kvasir ./result/ ./bs1_test/Kvasir/

       python3.7 Eval.py  ./ ./bs1_test/Kvasir/  ./result_bs1
      ```

      - 参数说明：
        - ./Kvasir：Kvasir数据路径
        - ./result：推理结果文件夹
        - ./bs1_test/Kvasir/：处理结果文件夹
        - ./: 当前路径
        - ./result_bs1：最终精度保存文件夹

      结果会打屏显示
   4. 性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型
        - --batchsize：模型batchsize
        - --loop: 循环次数



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度 | 性能 |
| -------- | ---------- | ------ | ---- | ---- |
|   310P3       |       1     |   Kvasir     | mDec:0.894<br>mIoU:0.836     |  170    |
|   310P3       |       4    |   Kvasir     | mDec:0.894<br>mIoU:0.836     |  425    |
