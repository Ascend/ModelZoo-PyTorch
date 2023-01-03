#  Segmenter 模型-推理指导

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

Segmenter提出了一种基于 Vision Transformer 的语义分割方法，可以更好地捕获上下文信息。

- 参考论文：

  [Segmenter: Transformer for Semantic Segmentation.](https://arxiv.org/pdf/2105.05633)

- 参考实现：

  ```
  url=https://github.com/rstrudel/segmenter
  commit_id=20d1bfad354165ee45c3f65972a4d9c131f58d53
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 768 x 768 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | output   | RGB_FP32 | batchsize x 3 x 768 x 768 | NCHW         |

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

2. 在同级目录下，获取开源模型代码

   ```
   git clone https://github.com/rstrudel/segmenter.git
   cd segmenter
   git checkout master
   git reset --hard 20d1bfad354165ee45c3f65972a4d9c131f58d53
   pip3 install -e .
   ```
   
3. 安装必要依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   - 下载标注数据：[gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1)
   - 下载原始图片：[leftImg8bit_travaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
   
   本模型用 cityscapes 的 val 集来验证模型精度。其中 cityscapes/gtFine/val 目录下存放的是标注后的 groundtruth 文件，此模型只用到后缀为 ‘gtFine_labelTrainIds.png’ 的图片；cityscapes/leftImg8bit/val 目录下存放原始的测试图片，包含三个城市共500张街景图片。该模型需要的groudtruth和原始图片的目录结构如下：
   
   ```
   cityscapes
   |-- gtFine
   |   |-- val
   |       |-- frankfurt
   |       |   |-- frankfurt_000000_000294_gtFine_labelTrainIds.png
   |       |   |-- frankfurt_000000_000576_gtFine_labelTrainIds.png
   |       |   |-- ......
   |       |-- lindau
   |       |   |-- lindau_000000_000019_gtFine_labelTrainIds.png
   |       |   |-- lindau_000001_000019_gtFine_labelTrainIds.png
   |       |   |-- ......
   |       `-- munster
   |           |-- munster_000000_000019_gtFine_labelTrainIds.png
   |           |-- munster_000001_000019_gtFine_labelTrainIds.png
   |           |-- ......
   `-- leftImg8bit
       |-- val
           |-- frankfurt
           |   |-- frankfurt_000000_000294_leftImg8bit.png
           |   |-- frankfurt_000000_000576_leftImg8bit.png
           |   |-- ......
           |-- lindau
           |   |-- lindau_000000_000019_leftImg8bit.png
           |   |-- lindau_000001_000019_leftImg8bit.png
           |   |-- ......
           `-- munster
               |-- munster_000000_000019_leftImg8bit.png
               |-- munster_000001_000019_leftImg8bit.png
               |-- ......
   ```
   
2. 数据预处理，将原始数据集转换为模型的输入数据。

   执行 segmenter_preprocess.py 脚本，完成数据预处理。

   ```
   touch gt_file.txt
   python3 segmenter_preprocess.py --cfg-path ${cfg_path} --data-root ${data_root} --bin-dir ${bin_dir} --gt-path gt_file.txt
   ```
   
   参数说明：
   
   - --cfg_path: 模型的配置文件。配置文件的获取方式参考**模型推理的步骤1**。
   - --data_root: cityscapes数据集所在的**父目录**。
   - --bin_dir: 指定预处理结果的存放目录。
   - --gt_file: 指定存放原始图片到标注图片映射的**文件**。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件和配置文件。

      配置文件 variant.yml 需要与模型权重 checkpoint.pth 放置于同一目录下。

      ```shell
      # 下载模型权重
      wget https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/cityscapes/seg_large_mask/checkpoint.pth
      # 下载配置文件
      wget https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/cityscapes/seg_large_mask/variant.yml
      ```

   2. 导出onnx文件。

      1. 使用segmenter_pytorch2onnx.py导出动态batch的onnx文件。

         ```
         python3 segmenter_pytorch2onnx.py -c ${checkpoint_path} -o ${onnx_path}
         ```

         参数说明：

         - -c, --checkpoint_path：权重文件路径
         - -o, --onnx_path：生成ONNX模型的保存路径

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
      atc --framework=5 --model=${onnx_path} --output=${om_path} \
      --input_format=NCHW --input_shape="input:${bs},3,768,768" \
      --log=error --soc_version=Ascend${chip_name} \
      --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
      ```

      运行成功后生成om模型文件。

      参数说明：
      
      - --model：为ONNX模型文件。
      - --framework：5代表ONNX模型。
      - --output：输出的OM模型。
      - --input_format：输入数据的格式。
      - --input_shape：输入数据的shape。
      - --log：日志级别。
      - --soc_version：处理器型号。
      - --optypelist_for_implmode：指定算子。
      - --op_select_implmode：配合optypelist_for_implmode参数，让指定算子走指定模式。

2. 开始推理验证。

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

      ```
      python3 -m ais_bench --model ${om_path} --input ${bin_dir} --output result --output_dirname result_bs${bs}
      ```
      
      参数说明：
      
      - --model：om模型路径。
      - --batchsize：批次大小。
      - --input：输入数据所在路径。
      - --output：推理结果输出路径。
      - --output_dirname：推理结果输出子文件夹。
      - --outfmt：推理结果输出格式
   
3. 精度验证。

   根据前处理生成的真值文件和离线推理生成的推理结果，计算模型精度。

   ```
   python3 segmenter_postprocess.py --result-dir ${result_dir} --gt-path ${gt_path} --metrics-path ${metrics_txt_path}
   ```

   参数说明：

   - --result_dir 存放推理结果的目录路径，这里为result/result_bs${bs}。
   - --gt_path 预处理生成的图片到标注的映射文件路径。
   - --metrics_path 指定一个路径用于记录模型指标。

4. 可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

   ```
   python3 -m ais_bench --model=${om-path} --loop=100 --batchsize=${bs}
   ```

   参数说明：

   - --model：om模型路径。
   - --batchsize：批次大小。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，Seg-L-Mask/16模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集     | 开源精度（mIoU）                                          | 参考精度（mIoU） |
| ----------- | ---------- | ---------- | --------------------------------------------------------- | ---------------- |
| Ascend310P3 | 1          | Cityscapes | [79.1%](https://github.com/rstrudel/segmenter#cityscapes) | 78.89%           |

| 芯片型号    | Batch Size | 参考性能（FPS） |
| ----------- | ---------- | --------------- |
| Ascend310P3 | 1          | 3.48            |
| Ascend310P3 | 4          | 2.91            |
| Ascend310P3 | 8          | 3.09            |
| Ascend310P3 | 16         | 3.09            |