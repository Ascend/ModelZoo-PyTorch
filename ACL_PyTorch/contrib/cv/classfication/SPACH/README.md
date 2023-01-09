#  SPACH 模型-推理指导

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

SPACH 是结合了卷积和Transformer模块的混合模型，应用于分类任务。本推理指导文档以其中的SPACH-Conv-MS-S配置为例。

- 参考论文：

  [A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP](https://arxiv.org/abs/2108.13002)

- 参考实现：

  ```
  url=http://github.com/microsoft/Spach
  commit_id=f69157d4e90fff88285766a4eabf51b29d772da3
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | output   | FP32     | batchsize x 1000 | ND           |

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

2. 获取开源模型代码

   ```
   git clone http://github.com/microsoft/Spach.git
   cd Spach
   git reset --hard f69157d4e90fff88285766a4eabf51b29d772da3
   cd ..
   ```
   
3. 安装必要依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   本模型使用[ImageNet](https://gitee.com/link?target=https%3A%2F%2Fimage-net.org%2Fdownload.php)验证集进行推理测试 ，用户自行获取数据集后，将文件解压并上传数据集到任意路径下。数据集目录结构如下所示：

   ```
   imageNet/
   |-- val
   |   |-- ILSVRC2012_val_00000001.JPEG
   |   |-- ILSVRC2012_val_00000002.JPEG
   |   |-- ILSVRC2012_val_00000003.JPEG
   |   ...
   |-- val_label.txt
   ...
   ```
   
2. 数据预处理，将原始数据集转换为模型的输入数据。

   执行 SPACH_preprocess.py 脚本，完成数据预处理。

   ```
   python3 ./SPACH_preprocess.py --src-path=${data_dir} --save-path=${save_dir}
   ```

   参数说明：

   - --data_dir：原数据集所在路径。
   - --save_dir：生成数据集二进制文件。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      获取权重文件 [spach_ms_conv_s.pth](https://github.com/microsoft/SPACH/releases/download/v1.0/spach_ms_conv_s.pth)，并将其放入当前工作目录。

   2. 导出onnx文件。

      1. 使用SPACH_pth2onnx.py导出动态batch的onnx文件。

         ```
         python3 SPACH_pth2onnx.py --input-path ${pth_file} --output-path ${onnx_file}
         ```

         参数说明：

         - --pth_file：权重文件。
         - --onnx_file：生成 onnx 文件。

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
      atc --model=spach_ms_conv_s.onnx --framework=5 --output=spach_ms_conv_s_bs${bs}\
       --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance\
       --log=error --input_format=NCHW --input_shape="input:${bs},3,224,224"\
       --output_type=FP16 --soc_version=Ascend${chip_name} --fusion_switch_file=fusion_switch.cfg
      ```

      运行成功后生成spach_bs${bs}.om模型文件。

      参数说明：
      
      - --model：为ONNX模型文件。
      - --framework：5代表ONNX模型。
      - --output：输出的OM模型。
      - --input_format：输入数据的格式。
      - --input_shape：输入数据的shape。
      - --log：日志级别。
      - --soc_version：处理器型号。
      - --fusion_switch_file：融合文件开关配置文件路径。

2. 开始推理验证。

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

      ```
      python3 -m ais_bench --model=spach_ms_conv_s_bs${bs}.om  --batchsize=${bs} \
      --input ${save_dir} --output result --output_dirname result_bs${bs} --outfmt TXT
      ```
      
      参数说明：
      
      - --model：om模型路径。
      - --batchsize：批次大小。
      - --input：输入数据所在路径。
      - --output：推理结果输出路径。
      - --output_dirname：推理结果输出子文件夹。
      - --outfmt：推理结果输出格式
   
3. 精度验证。

   调用SPACH_postprocess.py脚本与数据集标签val_label.txt比对，可以获得Accuracy数据。

   ```
   python SPACH_postprocess.py --txt-path=${result_dir} --label-path=${gt_file}
   ```

   参数说明：

   - --txt-path：推理结果所在路径，这里为 ./result/result_bs${bs}。
   - --label_path：真值标签文件val_label.txt所在路径。

4. 可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

   ```
   python3 -m ais_bench --model=spach_ms_conv_s_bs${bs}.om --loop=50 --batchsize=${bs}
   ```

   参数说明：

   - --model：om模型路径。
   - --batchsize：批次大小。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，SPACH-Conv-MS-S模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 开源精度（Acc@1）                                            | 参考精度（Acc@1） |
| ----------- | ---------- | -------- | ------------------------------------------------------------ | ----------------- |
| Ascend310P3 | 1          | ImageNet | [81.6%](https://github.com/microsoft/SPACH#main-results-on-imagenet-with-pretrained-models) | 81.5%             |

| 芯片型号    | Batch Size | 参考性能（FPS） |
| ----------- | ---------- | --------------- |
| Ascend310P3 | 1          | 325.89          |
| Ascend310P3 | 4          | 365.49          |
| Ascend310P3 | 8          | 462.96          |
| Ascend310P3 | 16         | 417.44          |
| Ascend310P3 | 32         | 388.13          |
| Ascend310P3 | 64         | 369.21          |