# Twins-SVT-L模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

  - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

   Twins-SVT 提出新的融合了局部-全局注意力的机制，可以类比于卷积神经网络中的深度可分离卷积 （Depthwise Separable Convolution），并因此命名为空间可分离自注意力（Spatially Separable Self-Attention，SSSA）。与深度可分离卷积不同的是，Twins-SVT 提出的空间可分离自注意力是对特征的空间维度进行分组，并计算各组内的自注意力，再从全局对分组注意力结果进行融合。
- 参考实现：

  ```
  url=https://github.com/Meituan-AutoML/Twins
  branch=master
  commit_id=4700293a2d0a91826ab357fc5b9bc1468ae0e987
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小       | 数据排布格式 |
  | -------- | -------- | ---------- | ------------ |
  | output  | FLOAT32  | batchsize x 1000  | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套       | 版本    | 环境准备指导                                                 |
  | ---------- | ------- | ------------------------------------------------------------ |
  | 固件与驱动 | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN       | 6.0.0 | -                                                            |
  | Python     | 3.7.5   | -                                                            |
  | Pytorch    | 1.7.0   | -                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/Meituan-AutoML/Twins
   ```

2. 整理代码结构

   ```
   mv Twins_postprocess.py Twins_preprocess.py Twins_pth2onnx.py Twins
   ```

3. 安装依赖

   ```
   pip install -r requirements.txt
   ```

   


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   该模型使用`ImageNet 2012`数据集进行精度评估，下载[ImageNet 2012数据集的验证集](https://gitee.com/link?target=https%3A%2F%2Fimage-net.org)， 原始数据集下载解压后直接是图像，没有按照类别区分，可参考[该链接](https://gitee.com/link?target=https%3A%2F%2Fzhuanlan.zhihu.com%2Fp%2F370799616)进行预处理，处理后的数据放到新建的`Twins/data/imagenet`目录下，文件结构如下：

   ```
   data
   └──imagenet
       └── val
           ├── n01440764
                 ├── ILSVRC2012_val_00000293.JPEG
                 ├── ILSVRC2012_val_00002138.JPEG
                 ……
                 └── ILSVRC2012_val_00048969.JPEG
           ├── n01443537
           ……
           └── n15075141

   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   在Twins路径下，执行centerface_pth_preprocess.py脚本，完成数据预处理。

   ```
   python Twins_preprocess.py --root_path data/imagenet/val --bin_path bin_path/
   ```

   - 参数说明：
     - --root_path:  原始数据验证集所在路径。
     - --bin_path:   输出的二进制文件保存路径。

   运行成功后，生成bin_path文件夹，bin_path目录下生成的是供模型推理的bin文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件 [alt_gvt_large.pth](https://drive.google.com/file/d/1um39wxIaicmOquP2fr_SiZdxNCUou8w-/view?usp=sharing)。放在Twins目录下。

   2. 导出onnx文件。

      使用alt_gvt_large.pth导出onnx文件。

      在Twins目录下，运行Twins_pth2onnx.py脚本。

      ```
      python Twins_pth2onnx.py --resume alt_gvt_large.pth --output alt_gvt_large.onnx
      ```

      - 参数说明：

        - --resume:  权重文件pth的路径。

        - --output:   输出的onnx模型文件路径。

      获得 alt_gvt_large.onnx 文件。

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
         atc --framework=5 --model=alt_gvt_large.onnx  --output=alt_gvt_large_bs${batch_size} --input_format=NCHW --input_shape="input:${batch_size},3,224,224" --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

   ​         运行成功后生成alt_gvt_large_bs${batch_size}.om模型文件。

   

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

      ```
      python -m ais_bench --model alt_gvt_large_bs${batch_size}.om --input bin_path/ --batchsize ${batch_size} --output result --output_dirname alt_gvt_large_bs${batch_size} --outfmt TXT
      ```

      -   参数说明：

           -   --model：om模型的路径。
           -   --input：输入模型的二进制文件路径。
           -   --output：推理结果输出目录。
           -   --output_dirname：推理结果输出的二级目录名。
           -   --batchsize：输入数据的batchsize。
           -   --outfmt：输出文件的类型。

      推理后的输出在当前目录result下。

      >**说明：** 
      >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见[参数详情](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer#%E5%8F%82%E6%95%B0%E8%AF%B4%E6%98%8E)。

   3. 精度验证。

      执行Twins_postprocess.py脚本完成后处理并得到精度。

      ```
      python Twins_postprocess.py --pred result/alt_gvt_large_bs${batch_size} --input_path data/imagenet/val
      ```

      - 参数说明：

        - --pred:  预测结果文件的路径。

        - --input_path:  输入原始数据集（.jpeg）的路径。

      运行成功后，程序会打印出模型的整体的精度指标：

      ```
      Accuracy of the network on the 50000 val images: 83.7%
      ```

      

   4. 性能验证。

      对于性能的测试，需要注意以下三点：

      - 测试前，请通过 npu-smi info 命令查看 NPU 设备状态，请务必在 NPU 设备空闲的状态下进行性能测试。
      - 为避免因测试持续时间太长而受到干扰，建议通过纯推理的方式进行性能测试。
      - 使用吞吐率作为性能指标，单位为 fps.

      > 吞吐率（throughput）：模型在单位时间（1秒）内处理的数据样本数。

      执行纯推理：

      ```
      python -m ais_bench --model alt_gvt_large_bs${batch_size}.om --loop 20 --batchsize ${batch_size}
      ```

      - 参数说明：
        - --model：om模型的路径
        - --batchsize：数据集batch_size的大小
        - --loop：推理循环的次数

      执行完纯推理命令，程序会打印出与性能相关的指标：

      ```
      [INFO] -----------------Performance Summary------------------
      [INFO] NPU_compute_time (ms): min = 431.9490051269531, max = 432.24200439453125, mean = 432.0568511962891, median = 432.0480041503906, percentile(99%) = 432.2307962036133
      [INFO] throughput 1000*batchsize(64)/NPU_compute_time.mean(432.0568511962891): 148.1286544184991
      ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

1、精度比对。
   
   自测了以下batchsize的精度，得到的精度没有差别，且与开源仓精度一致.

| Model         | Acc@1 | 基线精度Acc@1                                   |
| ------------ | ---------- | ------------------------------------------------ |
| ALTGVT-Large | 83.7%      | [83.7%](https://github.com/Meituan-AutoML/Twins) |



2、性能比对。

在 310P 设备上，当 batchsize 为 8 时模型性能最优，达 175.2209 fps.

| batchsize | 1        | 4        | 8        | 16       | 32       | 64       |
| --------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 310P      | 131.1062 | 129.6842 | 175.2209 | 161.2279 | 156.0500 | 148.1286 |
| 基线性能        | 61.9636  | 77.0060  | 79.461   | 83.6408  | 83.805   | 89.759   |
| 310P/基线性能   | 2.1158倍 | 1.6840倍 | 2.2051倍 | 1.9276倍 | 1.8620倍 | 1.6502倍 |