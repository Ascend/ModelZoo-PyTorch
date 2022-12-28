#  SE-ResNeXt101 模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)
  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)
  
- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

------

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

卷积神经网络CNN的核心是卷积操作，它通过融合局部感受野内的空间与通道信息来提取特征。大量已有的研究结果表明，提升特征层次结构中空间信息的编码，可以增强 CNN 的表示能力。在这项工作中，我们转而关注通道关系并提出了一种新颖的架构单元，我们将其称为 "Squeeze-and-Excitation"（SE）块，它通过显式构建通道之间的相互依赖关系来自适应地重新校准通道特征响应。将SE块堆叠形成 SENet 架构，可以有效地泛化不同的数据集。我们进一步证明，SE 块以很低的计算成本为现有最先进的 CNN 带来了显著的性能提升。以 Squeeze-and-Excitation Networks 为基础，我们在 ILSVRC 2017 分类竞赛中赢得冠军，并将 Top5 的错误率降低到 2.251%，相比 2016 年的获胜成绩提升了 25%。

- 参考论文：

  [[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)  ](https://arxiv.org/abs/1904.02877)

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmclassification/blob/v0.23.0/configs/_base_/models/seresnext101_32x4d.py
  tag=v0.23.0
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | image    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | class    | FP32     | batchsize x 1000 | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1** 版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fpies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.12.1  | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码

3. 安装必要依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集

   本模型推理项目使用 ILSVRC2012 数据集验证模型精度，请在 [ImageNet官网](http://image-net.org/) 自行下载ILSVRC2012数据集并解压，本模型将用到 ILSVRC2012_img_val.tar 验证集及 ILSVRC2012_devkit_t12.gz 中的val_label.txt数据标签。

   请按以下的目录结构存放数据：
   
   ```
   ├──ImageNet/
       ├──ILSVRC2012_img_val/
           ├──ILSVRC2012_val_00000001.JPEG
           ├──ILSVRC2012_val_00000002.JPEG
           ├──...
       ├──ILSVRC2012_devkit_t12/
           ├──val_label.txt
   ```
   
2. 数据预处理

   ```shell
   python3 Se_Resnext101_preprocess.py resnet ${data_dir} ${save_dir}
   ```

   参数说明：

   + 参数1：resnet 该模型数据预处理方式同 ResNet 网络，所以此处设置为resnet。
   + 参数2： 原始测试图片（.jpeg）所在目录的路径。
   + 参数3：指定一个目录用于存放生成的二进制（.bin）文件。

   运行成功后，每个图像对应生成一个二进制文件，存放于指定目录中。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      点击获取：[SE-ResNext101预训练pth权重文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/SE-ResNeXt101/PTH/state_dict.pth) 

   2. 导出onnx文件。

      1. 使用Se_Resnext101_pth2onnx.py导出动态batch的onnx文件。

         ```
         python3 Se_Resnext101_pth2onnx.py state_dict.pth se-resnext101.onnx
         ```

         参数说明：

         - --参数1：权重文件。
         - --参数2：生成 onnx 文件。

      2. 使用onnx-simplifier工具（使用方法参见[工具仓](https://github.com/daquexian/onnx-simplifier)）简化模型生成简化后的动态batch的onnx文件。

         ```shell
         python3 -m onnxsim se-resnext101.onnx se-resnext101-sim.onnx --overwrite-input-shape=-1,3,224,224
         ```

         参数说明

         + --overwrite-input-shape：输入数据的维度信息。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称（${chip_name}）。

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

   4. 执行ATC命令。

      ```
       # bs = [1, 4, 8, 16, 32, 64]
      atc --framework=5 \
          --model=se-resnext101-sim.onnx \
          --output=se-resnext_bs${bs} \
          --input_format=NCHW \
          --input_shape="image:${bs},3,224,224" \
          --log=error \
          --soc_version=Ascend${chip_name}
      ```
      
      运行成功后生成spnasnet_100_bs${bs}.om模型文件。
      
      参数说明：
      
      - --model：为ONNX模型文件。
      - --framework：5代表ONNX模型。
      - --output：输出的OM模型。
      - --input_format：输入数据的格式。
      - --input_shape：输入数据的shape。
      - --log：日志级别。
      - --soc_version：处理器型号。

2. 开始推理验证。

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

      ```
      python3 -m ais_bench --model se-resnext_bs${bs}.om --input ${save_dir} --output result --output_dirname result_bs${bs} --outfmt TXT --batchsize ${bs}
      ```
      
      参数说明：
      
      - --model：om模型路径。
      - --batchsize：批次大小。
      - --input：输入数据所在路径。
      - --output：推理结果输出路径。
      - --output_dirname：推理结果输出子文件夹。
      - --outfmt：推理结果输出格式
   
3. 精度验证。

   调用脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

   ```shell
   python3. Se_Resnext101_postprocess.py ${result_dir} ${gt_file} ./result.json
   ```

   参数说明：

   - 参数1：推理结果所在路径，这里为 ./result/result_bs${bs}。
   - 参数2：真值标签文件val_label.txt所在路径。
   - 参数3：精度计算结果保存文件。

4. 可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

   ```
   python3 -m ais_bench --model=se-resnext_bs${bs}.om --loop=50 --batchsize=${bs}
   ```

   参数说明：

   - --model：om模型路径。
   - --batchsize：批次大小。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，se-resnext101模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 开源精度（Acc@1）                                            | 参考精度（Acc@1） |
| ----------- | ---------- | -------- | ------------------------------------------------------------ | ----------------- |
| Ascend310P3 | 1          | ImageNet | [78.26%](https://github.com/open-mmlab/mmclassification/tree/master/configs/seresnet#imagenet-1k) | 78.24%            |

| 芯片型号    | Batch Size | 参考性能（FPS） |
| ----------- | ---------- | --------------- |
| Ascend310P3 | 1          | 555.81          |
| Ascend310P3 | 4          | 927.09          |
| Ascend310P3 | 8          | 844.82          |
| Ascend310P3 | 16         | 721.43          |
| Ascend310P3 | 32         | 421.14          |
| Ascend310P3 | 64         | 641.26          |