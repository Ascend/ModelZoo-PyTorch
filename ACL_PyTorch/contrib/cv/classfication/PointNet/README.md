# PointNet模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

`PointNet`由斯坦福大学提出，为点云神经网络先驱：提出了一种网络结构，可以直接从点云中学习特征。

- 参考实现：

  ```
  url=https://github.com/fxia22/pointnet.pytorch
  branch=master
  commit_id=f0c2430b0b1529e3f76fb5d6cd6ca14be763d975
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32  | batchsize x 3 x 2500      | ND           |

- 输出数据

  | 输出数据 | 大小            | 数据类型 | 数据排布格式 |
  | -------- | --------        | -------- | ------------ |
  | class    | batch_size x 16 | FLOAT32  | ND           |

# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                            | 版本    | 环境准备指导                                                                                          |
| ------------------------------------------------------------    | ------- | ------------------------------------------------------------                                          |
| 固件与驱动                                                      | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                            | 6.0.RC1 | -                                                                                                     |
| Python                                                          | 3.7.5   | -                                                                                                     |
| PyTorch                                                         | 1.5.0+ | -                                                                                                     |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master         # 切换到对应分支
   cd ACL_PyTorch/contrib/cv/classfication/PointNet              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   git clone https://gitee.com/zheng-wengang1/onnx_tools.git
   cd onnx_tools && git checkout cbb099e5f2cef3d76c7630bffe0ee8250b03d921
   cd ..
   ```

3. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/fxia22/pointnet.pytorch
   cd pointnet.pytorch
   git checkout f0c2430b0b1529e3f76fb5d6cd6ca14be763d975
   patch -p1 < ../modify.patch
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。

   本模型采用 [shapenetcore_partanno_segmentation_benchmark_v0](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip) ，解压到 `./data` 目录下（如没有则需要自己创建）。

   数据目录结构请参考：

   ```
   data
   └── shapenetcore_partanno_segmentation_benchmark_v0
    ├── 02591156
    ├── ...
    ├── README.txt
    ├── synsetoffset2category.txt
    └── train_test_split
   ```

2. 数据预处理。

   执行预处理脚本，生成数据集预处理后的bin文件:

   ```
   python3 pointnet_preprocess.py data/shapenetcore_partanno_segmentation_benchmark_v0 ./bin_file
   ```

   - 参数说明：第一个参数为数据集文件位置，第二个参数为输出文件位置。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      获取 [权重文件](https://pan.baidu.com/s/168Vk3C60iZOWrgGIBNAkjw)，提取码：lmwa 。得到 `checkpoint_79_epoch.pkl` 权重文件。

   2. 导出onnx文件。

      1. 使用脚本导出onnx文件。

         运行pointnet_pth2onnx.py脚本。

         ```
         # pth转换为ONNX
         python3 pointnet_pth2onnx.py --model checkpoint_79_epoch.pkl --output_file pointnet.onnx
         ```

         - 参数说明：第一个参数为模型配置文件，第二个参数是模型权重路径，第三个参数是导出onnx文件路径。

         获得pointnet.onnx文件。

     2. 优化onnx。

        ```
        # 以bs1为例
        python3 -m onnxsim pointnet.onnx pointnet_bs1_sim.onnx --input-shape="input:1,3,2500"
        python3 fix_conv1d.py pointnet_bs1_sim.onnx pointnet_bs1_sim_fixed.onnx
        ```

        获得pointnet_bs1_sim_fixed.onnx模型。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：**
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         回显如下：
         +-------------------|-----------------|------------------------------------------------------+
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
         # 以bs1为例
         atc --framework=5 --model=./pointnet_bs1_sim_fixed.onnx --output=./pointnet_bs1_fixed --input_shape="input:1,3,2500" --soc_version=${chip_name}  --log=error
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成模型文件pointnet_bs1_fixed.om。

2. 开始推理验证。

   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
        # 以bs1为例
        mkdir -p results/bs1
        python3 -m ais_bench --model ./pointnet_bs1_fixed.om --input ./bin_file --output ./results --output_dirname bs1 --batchsize 1 --outfmt TXT
        ```
        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --output_dirname：保存目录名。
             -   --device：NPU设备编号。
             -   --outfmt: 输出数据格式。
             -   --batchsize：推理模型对应的batchsize。


        推理后的输出默认在当前目录outputs/bs32下。

   3.  精度验证。

      调用pointnet_postprocess.py脚本与数据集标签比对，获得Accuracy数据。

      ```
      # 以bs1为例
      python3 pointnet_postprocess.py ./name2label.txt ./results/bs1 0
      ```

      -   参数说明：第一个参数为GT文件所在路径，第二个参数为推理结果路径，第三个参数为输出结果的key_idx（采用第0个输出作为最终结果）。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度参考下列数据:

| 模型     | pth精度     | 310精度     | 310P精度    |
| :------: | :------:    | :------:    | :------:    |
| PointNet | ACC: 97.42% | ACC: 97.35% | ACC: 97.35% |


推理性能：

| Model    | Batch Size | 310(FPS/Card) | 310(FPS/Card) | 基准(FPS/Card) |
|----------|------------|---------------|---------------|----------------|
| PointNet |          1 |           987 |       2374.11 |           1787 |
| PointNet |          4 |          1058 |       1980.89 |           2251 |
| PointNet |          8 |          1098 |       2176.16 |           2367 |
| PointNet |         16 |          1102 |       2256.15 |           2412 |
| PointNet |         32 |          1076 |       2217.42 |           2380 |
| PointNet |         64 |             - |       2205.12 |           2539 |
