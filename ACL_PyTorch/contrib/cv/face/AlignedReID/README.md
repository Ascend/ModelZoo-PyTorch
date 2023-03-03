# AlignedReID模型-推理指导

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

`AlignedReID` 算法提出了 `动态对齐（Dynamic Alignment）` 和 `协同学习（Mutual Learning）` 方法，该算法在部分数据集上首次在行人再识别问题上超越了人类表现。

- 参考实现：

  ```
  url=https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch.git
  branch=master
  commit_id=2e2d45450d69a3a81e15d18fe85c2eebbde742e4
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | image    | FLOAT32  | batchsize x 3 x 256 x 128 | NCHW         |

- 输出数据

  | 输出数据    | 大小                | 数据类型 | 数据排布格式 |
  | --------    | --------            | -------- | ------------ |
  | global_feat | batch_size x class  | FLOAT32  | ND           |
  | local_feat  | batch_size x class  | FLOAT32  | ND           |
  | logits      | batch_size x classs | FLOAT32  | ND           |

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
   cd ACL_PyTorch/contrib/cv/face/AlignedReID              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

3. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch.git 
   cd AlignedReID-Re-Production-Pytorch
   git reset --hard 2e2d45450d69a3a81e15d18fe85c2eebbde742e4
   patch -p1 < ../all.patch
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。

   获取 [Market1501](https://drive.google.com/drive/folders/1CaWH7_csm9aDyTVgjs7_3dlZIWqoBlv4)下载链接中的文件夹 `market1501` ，放在当前目录下，解压 `market1501` 中的 `images.tar` 压缩文件:

   ```
   cd market1501
   tar -xvf images.tar
   ```

   数据目录结构请参考：

   ```
   ├──market1501  
      │──images
      │──partitions.pkl
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行预处理脚本，生成数据集预处理后的bin文件:

   ```
   python3 AlignedReID_preprocess.py ./market1501/images ./prep_bin
   ```

   - 参数说明：第一个参数为数据集文件位置；第二个参数为输出文件位置。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取 [权重文件](https://pan.baidu.com/s/1SO0ONyCII7aq-MZ5W-pLcw)， 提取码：nrp2


   2. 导出onnx文件。

      1. 使用脚本导出onnx文件。

         运行AlignedReID_pth2onnx.py脚本。

         ```
         # pth转换为ONNX
         python3 AlignedReID_pth2onnx.py ./Market1501_AlignedReID_300_rank1_8441.pth ./AlignedReID_bs.onnx
         ```

         - 参数说明：第一个参数为模型权重文件路径，第二个参数为导出onnx文件路径。

         获得文件AlignedReID_bs.onnx。

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
         atc --framework=5 --model=AlignedReID_bs.onnx --output=AlignedReID_bs1 --input_format=NCHW --input_shape="image:1,3,256,128" --log=debug --soc_version=Ascend${chip_name} --out_nodes="Gemm_133:0;Reshape_127:0;Transpose_132:0"
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成模型文件AlignedReID_bs1.om。


2. 开始推理验证。

   1. 使用ais-bench工具进行推理。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        # 以bs1为例
        mkdir -p results/bs1
        python3 -m ais_bench --model AlignedReID_bs1.om --input ./prep_bin/ --output ./results --output_dirname bs1 --batchsize 1 --outfmt TXT
        ```
        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --output_dirname：保存目录名。
             -   --device：NPU设备编号。
             -   --outfmt: 输出数据格式。
             -   --batchsize：推理模型对应的batchsize。


        推理后的输出默认在当前目录outputs/bs1下。

   3.  精度验证。

      调用AlignedReID_acc_eval.py脚本与数据集标签比对，获得Accuracy数据。

      ```
      python3 AlignedReID_acc_eval.py ./results/bs1 ./market1501/partitions.pkl 1
      ```

      -   参数说明：第一个参数为推理结果路径，第二个参数为数据集对应的GT文件路径，第三个参数表示第二个输出结果为key，对应的idx为1

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度参考下列数据:

精度（re-ranking前）：

| accuracy | GPU    |    310 | 310p   |
| :------: | :----: | :----: | :----: |
| CMC1     | 81%    | 80.64% | 80.55% |


性能：

| 模型             |      310 |    基准性能 |    310P3 |
| :--------------: | :------: | :---------: | :------: |
| AlignedReID bs1  |   1441.7 | 1016.169286 |  1391.37 |
| AlignedReID bs4  | 1930.084 | 2238.839386 |  3939.99 |
| AlignedReID bs8  |  2160.46 | 2517.504524 |  4723.63 |
| AlignedReID bs16 |  2150.62 | 2737.996026 |  4241.01 |
| AlignedReID bs32 | 2208.228 | 3185.432007 |  5293.70 |
| AlignedReID bs64 |          |   3308.1090 |  3635.25 |
