# FairMOT模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

多目标跟踪(MOT)是计算机视觉中的一个长期目标,旨在估计视频中感兴趣对象的轨迹。问题的成功解决可以使许多应用受益,如视频分析和人机交互等。 现有的方法通常通过两个单独的模型来解决这个问题: 目标检测和重标识(Re-ID)。但是这两个网络不共享特征,故这些方法无法实时执行。随着多任务学习的成熟,人们对单网络MOT的关注增多。但由于目标检测和Re-ID的任务不公平,导致很多实验失败。因此比起之前两步(先检测后Re-ID)的跟踪算法,FairMOT完成检测与Re-ID共享网络参数,减少算法推理时间,速度大幅度提升。并在具体的实践中取得了很好的效果,FairMot网络采用DLA34作为backbone,使用6个混合的数据集进行训练,在MOT20数据集上进行推理。

- 参考实现: 

  ```
  url=https://github.com/ifzhang/FairMOT
  branch=master
  commit_id=4aa62976bde6266cbafd0509e24c3d98a7d0899f
  model_name=FairMOT
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 1088 x 608 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | batchsize x 1 x 152x 272 | FLOAT32  | ND           |
  | output2  | batchsize x 4 x 152x 272 | FLOAT32  | ND           |
  | output3  | batchsize x 128 x 152x 272 | FLOAT32  | ND           |
  | output4  | batchsize x 2 x 152x 272 | FLOAT32  | ND           |

# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.5.0   | -                                                            |
| 说明: Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本，torch版本仅可使用1.5.0（方便dcn的编译）。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。

   ```
      pip3 install -r requirment.txt # 需先注释掉 cython-bbox和yacs,等cython安装成功后再次安装
   ```
2. 安装DCN
   ```
      git clone -b pytorch_1.5 https://github.com/ifzhang/DCNv2.git
      cd DCNv2
      python3.7 setup.py build develop
      git reset 9f4254babcd162a809d165fa2430a780d14761f4 --hard
      patch -p1 < ../dcnv2.diff  
      cd ..
   ```

3. 下载开源模型代码
   ```
      git clone -b master https://github.com/ifzhang/FairMOT.git
      cd FairMOT
      git reset 2f36e7ebf640313a422cb7f07f93dc53df9b8d12 --hard
      patch -p1 < ../fairmot.diff
      cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   数据集名字: MOT17 
   获取地址: https://motchallenge.net/data/MOT17/
   使用 unzip MOT17.zip 进行解压
   数据如下目录
   ```
   MOT17
   ├──images
   │   ├── train
   │   ├── test
   ```

2. 数据预处理。
   创建数据目录
   目录如下:
   ```
   dataset
   ├── MOT17
   │   ├──images
   │   │   ├── train
   │   │   ├── test
   │   ├──label_with_ids
   │   │   ├── train
   │   │   ├── test
   ```
   构建label_with_ids
   ```
   mkdir dataset
   cd dataset
   cp -r ../MOT17/ .
   python3.7 FairMOT/src/gen_labels_16.py    # 用于生成label_with_ids,需修改路径为数据地址
   ```

   生成OM所需输入数据, --data_root=数据位置, --output_dir=输出预处理成功数据地址
   ```
   python3.7 ./fairmot_preprocess.py --data_root=./dataset --output_dir=./pre_dataset
   ```



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件,再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
      [fairmot_dla34.pth](https://drive.google.com/file/d/1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi/view?usp=sharing) 
   2. 导出onnx文件。

      1. 使用fairmot_pth2onnx.py导出onnx文件。

         运行fairmot_pth2onnx脚本。

         ```
            python3.7 fairmot_pth2onnx.py --input_file=fairmot_dla34.pth --output_file=fairmot.onnx
         ```

         获得fairmot.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明: ** 
         >该脚本中环境变量仅供参考,请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 
         回显如下: 
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
         atc --framework=5 --model=./fairmot.onnx --input_format=NCHW --input_shape="actual_input_1:1,3,608,1088" --output=./fairmot_bs1 --log=debug --soc_version=Ascend${chip_name}
         ```

         - 参数说明: 

           -   --model: 为ONNX模型文件。
           -   --framework: 5代表ONNX模型。
           -   --output: 输出的OM模型。
           -   --input\_format: 输入数据的格式。
           -   --input\_shape: 输入数据的shape。
           -   --log: 日志级别。
           -   --soc\_version: 处理器型号，  ${chip_name}可通过npu-smi info指令查看。

           运行成功后生成<u>***fairmot_bs1.om***</u>模型文件。



2. 开始推理验证。

a.  安装ais_bench推理工具。

   请点击本链接进行安装ais_bench推理工具，以及查看具体使用方法(https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)。  

b.  执行推理（支持bs1到bs8,其他bs模型太大无法跑出）。

    python3.7 -m ais_bench  --model fairmot_bs1.om --input pre_dataset --device 0 -o ./ --outfmt BIN --batchsize 1

   - 参数说明: 
      -   --model: om文件路径。
      -   --device: NPU设备编号。
      -   --input: 输入数据地址。
      -   -o: 输出地址,默认会使用时间信息作为名字
      -   --outfmt: 输出格式(后续脚本只支持bin格式)
		...

        > **说明: ** 
        >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见。

c.  精度验证。

    python3.7 ./fairmot_postprocess.py --data_dir=./dataset  --input_root=./2022xxx > bs_1_result.log  # 调用脚本与数据集标签比对,可以获得MOTA指标,结果保存在bs_1_result.log中。
    python3.7 test/parse.py bs_1_result.log

   - 参数说明: 
      -   --data_dir: 数据路径。
      -   --input_root: 上一步生成的结果文件。
   使用test/parse.py输出MOTA值。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算,性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|   310P3   |         1        |   MOT17    |     83.8%       |        10.82        |
|   310P3   |         4        |   MOT17    |      -      |        11.24        |
|   310P3   |         8        |   MOT17    |      -      |        11.24        |
