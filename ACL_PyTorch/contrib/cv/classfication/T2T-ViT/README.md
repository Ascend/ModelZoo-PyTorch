#  T2T-ViT 模型-推理指导

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

balabala

- 参考论文：

  [Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://arxiv.org/abs/2101.11986)

- 参考实现：

  ```
  url=https://github.com/yitu-opensource/T2T-ViT
  branch=main
  commit_id=0f63dc9558f4d192de926504dbddfa1b3f5db6ca
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

1. 获取开源模型代码。

   ```
   git clone https://github.com/yitu-opensource/T2T-ViT.git
   cd T2T-ViT
   git reset --hard 0f63dc9558f4d192de926504dbddfa1b3f5db6ca
   ```

2. 获取本仓源码，并放置于第1步获得的T2T-ViT目录下。

3. 安装必要依赖。

   ```
   pip3 install -r requirements.txt
   ```

4. 源码改动。

   1. models/token_performer.py文件打补丁：

      ```
      patch -p1 models/token_performer.py token_performer.patch
      ```

      - 备注：由于OM模型中Einsum算子低精度计算（float16）会放大误差，导致精度问题。在转OM时使用--keep_dtype参数，尝试让Einsum算子保持原精度(float32)计算，但Einsum算子前面的TransData算子又会使此操作失效。所以定位到Einsum算子在模型源码中的位置，对其进行等价替换后，重新转ONNX，转OM时再使用--keep_dtype参数，TransData算子被消除，--keep_dtype参数生效。

   2. 进入python第三方库timm/data文件夹内：

      ```
      patch -p1 loader.py ${patch_path}
      ```

      - ${patch_path} 为 loaders.patch 文件路径。

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   本模型使用[ImageNet](https://gitee.com/link?target=https%3A%2F%2Fimage-net.org%2Fdownload.php)验证集进行推理测试 ，用户自行获取数据集后，将文件解压并上传数据集到任意路径下。数据集目录结构如下所示：

   ```
   │imagenet/
   ├──val/
   │  ├── n01440764
   │  │   ├── ILSVRC2012_val_00000293.JPEG
   │  │   ├── ILSVRC2012_val_00002138.JPEG
   │  │   ├── ......
   │  ├── ......
   ```
   
2. 数据预处理，将原始数据集转换为模型的输入数据。

   运行T2T_ViT_preprocess.py预处理脚本，生成数据集预处理后的bin文件。

   ```
   python3 T2T_ViT_preprocess.py --data-dir ${data_dir} --out-dir ${prep_bin} --gt-path ${gt_path}
   ```

   参数说明：

   - --data-dir：数据集路径。
   - --out-dir：指定保存bin文件的路径，比如 prep_bin。
   - --gt-path：生成npy真值文件路径， 比如 label.npy。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      ```
      wget https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.5_T2T_ViT_14.pth.tar
      ```

   2. 导出onnx文件。

      1. 使用T2T_ViT_pth2onnx.py导出动态batch的onnx文件。

         ```
         python3 T2T_ViT_pth2onnx.py --pth-path 81.5_T2T_ViT_14.pth.tar --onnx-path T2T_ViT.onnx
         ```

         参数说明：

         - --pth-path：Pytorch模型文件路径。
         - --onnx-path：ONNX模型文件保存路径。

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

   4. 安装 [auto_optimizer工具](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer#%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B)， 生成指定算子精度模式的配置文件。

      ```
      # 安装依赖
      git clone https://gitee.com/ascend/msadvisor.git
      cd msadvisor/auto-optimizer
      pip install -r requirements.txt
      
      # 运行脚本生成cfg文件
      python3 gen_cfg.py T2T_ViT.onnx keep_dtype.cfg
      ```

      参数说明：

      - --参数1：为ONNX模型文件。
      - --参数2：生成的配置文件。

   5. 执行ATC命令。

      ```
      # bs = [1, 4, 8, 16, 32, 64]
      atc --framework=5 --model=T2T_ViT.onnx --output=T2T_ViT_bs${bs} --input_format=NCHW --input_shape="image:${bs},3,224,224" --log=error --soc_version=Ascend${chip_name} --keep_dtype=keep_dtype.cfg
      ```

      运行成功后生成T2T_ViT_bs${bs}.om模型文件。

      参数说明：

      - --model：为ONNX模型文件。
      - --framework：5代表ONNX模型。
      - --output：输出的OM模型。
      - --input_format：输入数据的格式。
      - --input_shape：输入数据的shape。
      - --log：日志级别。
      - --soc_version：处理器型号。
      - --keep_dtype：指定算子精度模式的配置文件。

2. 开始推理验证。

   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

      ```
      python3 -m ais_bench --model=T2T_ViT_bs${bs}.om  --batchsize=${bs} \
      --input ${prep_bin} --output result --output_dirname result_bs${bs} --outfmt BIN
      ```
      
      参数说明：
      
      - --model：om模型路径。
      - --batchsize：批次大小。
      - --input：输入数据所在路径。
      - --output：推理结果输出路径。
      - --output_dirname：推理结果输出子文件夹。
      - --outfmt：推理结果输出格式
   
3. 精度验证。

   调用T2T_ViT_postprocess.py脚本与真值标签比对，可以获得精度数据。

   ```
   python3 T2T_ViT_postprocess.py --result-dir ${result_dir} --gt-path ${gt_path}
   ```

   参数说明：

   - --result-dir：生成推理结果所在路径，这里为result/result_bs1
   - --gt-path：预处理时生成的标签数据文件路径，这里为label.npy

4. 可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

   ```
   python3 -m ais_bench --model=T2T_ViT_bs${bs}.om --loop=50 --batchsize=${bs}
   ```

   参数说明：

   - --model：om模型路径。
   - --batchsize：批次大小。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，T2T_ViT模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 开源精度（Acc@1）                                            | 参考精度（Acc@1） |
| ----------- | ---------- | -------- | ------------------------------------------------------------ | ----------------- |
| Ascend310P3 | 1          | ImageNet | [81.5%](https://github.com/yitu-opensource/T2T-ViT#2-t2t-vit-models) | 81.4%             |

| 芯片型号    | Batch Size | 参考性能（FPS） |
| ----------- | ---------- | --------------- |
| Ascend310P3 | 1          | 163.14          |
| Ascend310P3 | 4          | 170.73          |
| Ascend310P3 | 8          | 194.66          |
| Ascend310P3 | 16         | 189.98          |
| Ascend310P3 | 32         | 180.15          |
| Ascend310P3 | 64         | 174.90          |