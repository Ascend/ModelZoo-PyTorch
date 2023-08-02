#  Squeezenet1_1模型-推理指导

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

Squeezenet的设计采用了卷积替换、减少卷积通道数和降采样操作后置等策略，旨在在不大幅降低模型精度的前提下，最大程度的提高运算速度。

- 参考论文：

  [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)

- 参考实现：

  ```
  https://github.com/pytorch/vision/blob/v0.14.0/torchvision/models/squeezenet.py#L193
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | image    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | class    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |

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

2. 安装依赖。

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

   执行 squeezenet1_1_preprocess.py 脚本，完成数据预处理。

   ```
   python3 squeezenet1_1_preprocess.py ${data_dir} ${save_dir} 
   ```

   参数说明：

   - --data_dir：原数据集所在路径。
   - --save_dir：生成数据集二进制文件。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [squeezenet1_1-f364aa15.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Squeezenet1_1/PTH/squeezenet1_1-f364aa15.pth)

   2. 导出onnx文件。

      1. 使用squeezenet1_1_pth2onnx.py导出动态batch的onnx文件。

         ```
         python3 squeezenet1_1_pth2onnx.py ${pth_file} ${onnx_file}
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
       atc --model=${onnx_file} --framework=5 --output=squeezenet1_1_bs${bs} \
       --input-shape="image:${bs},3,224,224" --log=error --soc_version=Ascend${chip_name} --enable_small_channel=1 --insert_op_conf=aipp.config
      ```

      运行成功后生成squeezenet1_1_bs${bs}.om模型文件。

      参数说明：

      - --model：为ONNX模型文件。
      - --framework：5代表ONNX模型。
      - --output：输出的OM模型。
      - --input_format：输入数据的格式。
      - --input_shape：输入数据的shape。
      - --log：日志级别。
      - --soc_version：处理器型号。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```
      mkdir result
      python3 -m ais_bench --model=squeezenet1_1_bs${bs}.om  --batchsize=${bs} \
      --input ${save_dir} --output result --output_dirname result_bs${bs} --outfmt TXT
      ```

      参数说明：

      - --model：om模型路径。
      - --batchsize：批次大小。
      - --input：输入数据所在路径。
      - --output：推理结果输出路径。
      - --output_dirname：推理结果输出子文件夹。
      - --outfmt：推理结果输出格式。

3. 精度验证。

   调用脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

   ```
   python3 squeezenet1_1_postprocess.py ${result_dir} ${gt_file} result.json
   ```

   参数说明：

   - --result_dir：推理结果所在路径，这里为 ./result/result_bs${bs}。
   - --gt_file：真值标签文件val_label.txt所在路径。

4. 可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

   ```
   python3 -m ais_bench --model=squeezenet1_1_bs${bs}.om --loop=50 --batchsize=${bs}
   ```

   参数说明：

   - --model：om模型路径。
   - --batchsize：批次大小。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，Squeezenet1_1模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 开源精度                                              | 精度指标1（Acc@1） | 精度指标2（Acc@5） |
| ----------- | ---------- | -------- | ----------------------------------------------------- | ------------------ | ------------------ |
| Ascend310P3 | 1          | ImageNet | [链接](https://pytorch.org/vision/stable/models.html) | 57.32%             | 80.06%             |
| Ascend310B1 | 1          | ImageNet | [链接](https://pytorch.org/vision/stable/models.html) | 57.32%             | 80.06%             |

| Batch Size | 310P3 | 310B1 |
| ---------- | ----------- | ---------- |
| 1          | 5797.1     | 3146.63 |
| 4          | 15289.3    | 4576.65 |
| 8          | 21301.5    | 3871.84 |
| 16         | 18339.9    | 3629.88 |
| 32         | 15468.8    | 3544.9 |
| 64         | 14051.23    | 822.58 |
| **最优性能** | **21301.5** | **4576.65** |