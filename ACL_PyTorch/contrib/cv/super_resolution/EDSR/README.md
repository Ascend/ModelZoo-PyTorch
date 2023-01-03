# EDSR模型-推理指导

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

论文通过提出EDSR模型移除卷积网络中不重要的模块并且扩大模型的规模，使网络的性能得到提升。

- 参考实现：

  ```
  url=https://github.com/sanghyun-son/EDSR-PyTorch.git
  branch=master
  commit_id=9d3bb0ec620ea2ac1b5e5e7a32b0133fbba66fd2
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- | -------- | -------------------------   | ------------ |
  | input    | FLOAT32  | batchsize x 3 x 1020 x 1020 | NCHW         |

- 输出数据

  | 输出数据 | 大小                         | 数据类型 | 数据排布格式 |
  | -------- | --------                     | -------- | ------------ |
  | output   | batch_size x 3 x 2040 x 2040 | FLOAT32  | NCHW        |


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
   cd ACL_PyTorch/contrib/cv/super_resolution/EDSR              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

3. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/sanghyun-son/EDSR-PyTorch.git
   cd EDSR-PyTorch && git checkout 9d3bb0ec
   patch -p1 < ../edsr.diff
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。

   本模型采用[DIV2K官网](https://data.vision.ee.ethz.ch/cvl/DIV2K/)的100张验证集进行测试
   其中，低分辨率图像(LR)采用bicubic x2处理(Validation Data Track 1 bicubic downscaling x2 (LR images))，高分辨率图像(HR)采用原图验证集(Validation Data (HR images))。

   数据目录结构请参考：

   ```
   ├── DIV2K              
         ├──HR  
              │──图片1
              │──图片2
              │   ...
         ├──LR
              │──图片1
              │──图片2
              │   ...       
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行预处理脚本，生成数据集预处理后的bin文件:

   ```
   python3 edsr_preprocess.py -s DIV2K/LR -d ./prep_data
   ```

   - 参数说明：

     -s: 数据集文件位置。

     -d：输出文件位置。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      下载地址[pth权重文件](https://cv.snu.ac.kr/research/EDSR/model_pytorch.tar)，解压得到 `edsr_x2.pt` 文件。

   2. 导出onnx文件。

      1. 使用脚本导出onnx文件。

         运行edsr_pth2onnx.py脚本。

         ```
         # pth转换为ONNX
         mkdir -p models/onnx
         python3 edsr_pth2onnx.py --pth edsr_x2.pt --onnx models/onnx/edsr_x2_dynamic.onnx --size 1020
         ```

         - 参数说明：

           --pth: 模型权重。

           --onnx：导出onnx文件路径。

           --size: 输入图片尺寸大小。

         获得models/onnx/edsr_x2.onnx文件。

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
         atc --model=models/onnx/edsr_x2_dynamic.onnx --framework=5 --output=models/om/edsr_x2_bs1 --input_format=NCHW --input_shape="input.1:1,3,1020,1020" --log=debug --soc_version=Ascend${chip_name} --fusion_switch_file=switch.cfg
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --fusion\_switch\_file：融合规则配置。

           运行成功后生成edsr_x2_bs1.om模型文件。



2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        # 以bs1为例
        mkdir -p results/bs1
        python3 -m ais_bench --model ./models/om/edsr_x2_bs1.om --input ./prep_data/bin --output ./results --output_dirname bs1 --batchsize 1
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

      调用edsr_postprocess.py脚本与数据集标签比对，获得Accuracy数据。

      ```
      python3 edsr_postprocess.py --res ./results/bs1 --HR ./DIV2k/HR
      ```

      -   参数说明：

        --res：推理结果所在路径。

        --HR：GT数据文件所在路径。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度参考下列数据:

| device |   ACC |
|--------|-------|
| 基准   | 34.6% |
| 310    | 34.6% |
| 310P   | 34.6% |

性能参考下列数据(memory限制只支持到bs8)。

| 模型     | 基准性能  | 310性能  | 310P3性能 |
| :------: | :------:  | :------: |  :------: |
| EDSR bs1 | 6.0501fps | 4.911fps |    7.9055 |
| EDSR bs4 | 5.6783fps | 4.6448   |    7.8842 |
| EDSR bs8 | 5.7391fps | -        |    7.0100 |

