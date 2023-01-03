# 3D_HRNet模型-推理指导

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

`High-Resoultion Net(HRNet)` 由微软亚洲研究院和中科大提出，发表在CVPR2019。HRNet的网络在CV领域，越来越得到关注，因为很多用HRNet作为骨架网络的方案在目标检测、分类、分割、人体姿态估计等领域均取得瞩目的成绩。本仓为HRNet应用于分割的样例。

- 参考实现：

  ```
  url=https://github.com/HRNet/HRNet-Semantic-Segmentation.git
  branch=master
  commit_id=0bbb2880446ddff2d78f8dd7e8c4c610151d5a51
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- | -------- | -------------------------   | ------------ |
  | image    | FLOAT32  | batchsize x 3 x 1024 x 2048 | NCHW         |

- 输出数据(TODO)

  | 输出数据 | 大小                        | 数据类型 | 数据排布格式 |
  | -------- | --------                    | -------- | ------------ |
  | output1  | batch_size x 19 x 256 x 512 | FLOAT32  | NCHW         |
  | output2  | batch_size x 19 x 256 x 512 | FLOAT32  | NCHW         |

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
   cd ACL_PyTorch/contrib/cv/segmentation/3D_HRNet              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   # 改图依赖
   git clone https://gitee.com/Ronnie_zheng/MagicONNX.git MagicONNX
   cd MagicONNX && git checkout dev
   pip3 install . && cd ..
   ```

3. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/HRNet/HRNet-Semantic-Segmentation.git
   cd HRNet-Semantic-Segmentation
   git reset 0bbb2880446ddff2d78f8dd7e8c4c610151d5a51 --hard
   patch -p1 < ../HRNet.patch
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。

   本模型采用[cityscpaes数据集](https://www.cityscapes-dataset.com)，解压到 `./data` （如没有则需要手动创建）。

   数据目录结构请参考(TODO)：

   ```
   data
   |-- cityscapes
      |-- gtFine
      |    |-- test
      |    |-- train
      |    |-- val
      |-- leftImg8bit
           |-- test
           |-- train
           |-- val
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行预处理脚本，生成数据集预处理后的bin文件:

   ```
   mkdir prep_dataset
   python3 HRNet_preprocess.py --src_path=./data --save_path=./prep_dataset
   ```

   - 参数说明：

     --src_path: 数据集文件位置。

     --save_path：输出文件位置。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      下载[pth模型](https://pan.baidu.com/s/1xGZFEQTjhsP4sj0Lvf_sXg)到推理目录下（提取码: e5ci）。



   2. 导出onnx文件。

      1. 使用脚本导出onnx文件。

         运行HRNet_pth2onnx.py脚本。

         ```
         # pth转换为ONNX
         python3 HRNet_pth2onnx.py --pth hrnet.pth --save_path hrnet.onnx
         ```

         - 参数说明：

           --pth: 模型权重路径。

           --save_path：导出onnx文件路径。

         获得hrnet.onnx文件。

     2. 优化onnx。

        运行performance_optimize_resize.py脚本优化：

        ```
        python3 performance_optimize_resize.py hrnet.onnx hrnet.onnx
        ```

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
         atc --framework=5 --model=hrnet.onnx --output=hrnet_bs1 --input_format=NCHW --input_shape="image:1,3,1024,2048" --log=debug --soc_version=Ascend${chip_name} --out_nodes="Conv_1380:0;Conv_1453:0"
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成模型文件hrnet_bs1.om。


2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        # 以bs1为例
        mkdir -p results/bs1
        python3 -m ais_bench --model hrnet_bs1.om --input ./prep_dataset/ --output ./results --output_dirname bs1 --batchsize 1
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

      调用HRNet_postprocess.py脚本与数据集标签比对，获得Accuracy数据。

      ```
      python3 HRNet_postprocess.py --res_path results/bs1 --data_path data/cityscapes/gtFine/val/ --save_path ./result_bs1.json
      ```

      -   参数说明：

        --res_path：推理结果所在路径。
        --data_path：数据集所在路径。
        --save_path：最后结果保存路径。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度参考下列数据:

说明：内存限制该模型支持bs1/bs4

| 3D HRNet模型 | gpu吞吐率 | 310吞吐率 | 310P吞吐率 | 目标精度 | 310精度 | 310P精度 |
|--------------|-----------|-----------|------------|----------|---------|----------|
| bs1          | 5.85fps   | 4.90fps   | 9.55fps    |    81.6% |  80.85% |   80.83% |
| bs4          | 5.75fps   | 4.78fps   | 8.07fps |    81.6% |  80.85% |   80.83% |
