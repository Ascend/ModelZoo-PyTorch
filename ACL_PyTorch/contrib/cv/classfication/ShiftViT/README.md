#  ShiftViT 模型-推理指导

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

在Transformer基础上，提出了更加简单的 shift block 来代替 self-attention，在图像分类、检测、分割等任务上效果显著。本文验证的是开源仓中的 Shift-T/light 模型。

- 参考论文：

  [When Shift Operation Meets Vision Transformer: An Extremely Simple Alternative to Attention Mechanism](https://arxiv.org/abs/2201.10801)

- 参考实现：

  ```
  url=https://github.com/microsoft/SPACH
  commit_id=20d1bfad354165ee45c3f65972a4d9c131f58d53
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据       | 数据类型 | 大小                      | 数据排布格式 |
  | -------------- | -------- | ------------------------- | ------------ |
  | actual_input_1 | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | output1  | FP32     | batchsize x 1000 | ND           |

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

1. 获取开源模型代码。

   ```
   git clone https://github.com/microsoft/SPACH.git
   cd SPACH
   git checkout main
   git reset --hard f69157d4e90fff88285766a4eabf51b29d772da3
   ```

2. 获取本仓源码，置于SPACH目录下。

3. 安装必要依赖。

   ```
   pip3 install -r requirements.txt
   ```

4. 安装改图依赖 auto_optimizer，安装和使用详情参考[链接](https://gitee.com/sibylk/msadvisor/tree/master/auto-optimizer)。

   ```
   git clone https://gitee.com/ascend/msadvisor.git
   cd auto-optimizer
   pip3 install -r requirements.txt
   python3 setup.py install
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   本离线推理项目使用 ILSVRC2012 数据集（ImageNet-1k）的验证集进行精度验证。从 http://image-net.org/ 下载数据集并解压，其中 val 的目录结构遵循 [torchvision.datasets.ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#torchvision.datasets.ImageFolder) 的标准格式：

   ```
   /path/to/imagenet/
   ├──val/
   │  ├── n01440764
   │  │   ├── ILSVRC2012_val_00000293.JPEG
   │  │   ├── ILSVRC2012_val_00002138.JPEG
   │  │   ├── ......
   │  ├── ......
   ```

2. 数据预处理，将原始数据集转换为模型的输入数据。

   执行 shiftvit_preprocess.py 脚本，完成数据预处理。

   ```shell
   python3 shiftvit_preprocess.py --data-root ${data_dir} --save-dir ${save_dir} --gt-path val-gt.npy
   ```

   参数说明：

   - --data-root：i数据集所在路径
   - --save-dir：存放预处理后数据文件的路径
   - --gt-path：真值标签路径，存放图片的分类标签

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      ```shell
      # 下载 Pytorch 模型权重
      wget https://github.com/microsoft/SPACH/releases/download/v1.0/shiftvit_tiny_light.pth 
      
      # timm 包内的 models/layers/helpers.py 与 torch 1.10.0 存在冲突，需修改
      patch -p0 path/to/envs/spach/lib/python3.7/site-packages/timm/models/layers/helpers.py helpers.patch
      ```

   2. 导出onnx文件。

      1. 为提升模型性能，对模型源码 shift_feat 方法中 feature 数值更新操作进行等价替换

         ```shell
         patch -p0 models/shiftvit.py shiftvit.patch
         ```

      2. 生成ONNX模型

         ```shell
         python3 shiftvit_pytorch2onnx.py -c shiftvit_tiny_light.pth -o shiftvit1.onnx
         ```

         参数说明：

         - -c, --checkpoint-path：PyTorch 权重文件路径
         - -o, --onnx-path：ONNX 模型的保存路径
         - -v, --opset-version：ONNX 算子集版本, 默认 12

      1. 为提升模型精度，对 ONNX 模型进行修改

         ```shell
         python3 modify_onnx.py -i shiftvit1.onnx -o shiftvit2.onnx
         ```

         参数说明：

         - -i，--input-onnx：原始 ONNX 模型的路径
         - -o，--output-onnx：修改后 ONNX 模型的保存路径

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
          --model=shiftvit2.onnx \
          --output=shiftvit2-bs${bs} \
          --input_format=NCHW \
          --input_shape="input:${bs},3,224,224" \
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
      python3 -m ais_bench --model=shiftvit2-bs${bs}.om --batchsize=${bs} \
      --input ${save_dir} --output result --output_dirname result_bs${bs}
      ```
      
      参数说明：
      
      - --model：om模型路径。
      - --batchsize：批次大小。
      - --input：输入数据所在路径。
      - --output：推理结果输出路径。
      - --output_dirname：推理结果输出子文件夹。
      - --outfmt：推理结果输出格式
   
3. 精度验证。

   调用脚本与数据集真值标签比对，可以获得精度结果。

   ```
   python3 shiftvit_postprocess.py --result-dir ${result_dir} --gt-path ${gt_file}
   ```

   参数说明：

   - --result-dir：推理结果所在路径，这里为 ./result/result_bs${bs}。
   - --gt-path：数据预处理时生成的真值标签文件，这里为val-gt.npy。

4. 可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

   ```
   python3 -m ais_bench --model=shiftvit2-bs${bs}.om --loop=50 --batchsize=${bs}
   ```

   参数说明：

   - --model：om模型路径。
   - --batchsize：批次大小。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，spnasnet_100模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 开源精度（Acc@1）                                            | 参考精度（Acc@1） |
| ----------- | ---------- | -------- | ------------------------------------------------------------ | ----------------- |
| Ascend310P3 | 1          | ImageNet | [79.4%](https://github.com/microsoft/SPACH#main-results-on-imagenet-with-pretrained-models) | 79.30%            |

| 芯片型号    | Batch Size | 性能（FPS） |
| ----------- | ---------- | ----------- |
| Ascend310P3 | 1          | 338.64      |
| Ascend310P3 | 4          | 706.70      |
| Ascend310P3 | 8          | 842.25      |
| Ascend310P3 | 16         | 825.09      |
| Ascend310P3 | 32         | 801.01      |
| Ascend310P3 | 64         | 809.59      |