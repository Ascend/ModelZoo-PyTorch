#  WDSR 模型-推理指导

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

Wdsr是对EDSR进行改进，去除了冗余的卷积层，同时也改造了resblock。从而可以在同样计算开销的前提下，能够有更好的性能。

- 参考论文：

  [*Jiahui Yu, Yuchen Fan, Jianchao Yang, Ning Xu, Zhaowen Wang, Xinchao Wang, Thomas Huang.**Wide Activation for Efficient and Accurate Image Super-Resolution.(2018)*](https://arxiv.org/abs/1808.08718)

- 参考实现：

  ```
  url=https://github.com/ychfan/wdsr
  commit_id=b78256293c435ef34e8eab3098484777c0ca0e10
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                       | 数据排布格式 |
  | -------- | -------- | -------------------------- | ------------ |
  | image    | RGB_FP32 | batchsize x 3 x 1020 x1020 | NCHW         |

- 输出数据

  | 输出数据  | 数据类型 | 大小                        | 数据排布格式 |
  | --------- | -------- | --------------------------- | ------------ |
  | out_image | RGB_FP32 | batchsize x 3 x 2040 x 2040 | NCHW         |

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
   git clone https://github.com/ychfan/wdsr.git -b master 
   cd wdsr
   git reset --hard b78256293c435ef34e8eab3098484777c0ca0e10
   cd ..
   ```
   
3. 安装必要依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   下载[DIV2K数据集](https://data.vision.ee.ethz.ch/cvl/DIV2K/)`Validation Data (HR images)`和`Validation Data Track 1 bicubic downscaling x2 (LR images)`两个压缩包，新建data/DIV2K文件夹，将两个压缩包解压至该文件夹中。

   ```
   data/DIV2K/
   |-- DIV2K_valid_HR
   |   |-- 0801.png
   |   |-- 0802.png
   |   |-- 0803.png
   |   |-- ......
   |-- DIV2K_valid_LR_bicubic
   |   `-- X2
   |       |-- 0801x2.png
   |       |-- 0802x2.png
   |       |-- 0803x2.png
   |       |-- ...
   ```

2. 数据预处理，将原始数据集转换为模型的输入数据。

   执行 wdsr_preprocess.py 脚本，完成数据预处理。

   ```
   python3 wdsr_preprocess.py --lr_path ./data/DIV2K/DIV2K_valid_LR_bicubic/X2/ --hr_path ./data/DIV2K/DIV2K_valid_HR/ --save_lr_path ${prep_data}  --width 1020 --height 1020 --scale 2 
   ```

   参数说明：

   - --参数1：为低分辨率数据集相对路径。
   - --参数2：为高分辨率数据集的相对路径。
   - --参数3：为生成数据集文件的保存路径。
   - --参数4：缩放大小。

   运行成功后，在当前目录生成DIV2K_valid_LR_bicubic_bin/X2/数据集。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件 epoch_30.pth。

      ```
      wget https://github.com/ychfan/wdsr/files/4176974/wdsr_x2.zip
      unzip wdsr_x2.zip
      ```

   2. 导出onnx文件。

      1. 使用wdsr_pthr2onnx.py导出动态batch的onnx文件。

         ```
         python3 wdsr_pth2onnx.py --ckpt epoch_30.pth --model wdsr --output_name wdsr.onnx --scale 2
         ```

         参数说明：

         - --参数1：权重文件。
         - --参数2：要导入的模型名称。
         - --参数3：输出onnx文件的名称。
         - --参数4：缩放大小。

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
       atc --framework=5 --model=wdsr.onnx --output=wdsr_bs${bs} --input_format=NCHW --input_shape="image:${bs},3,1020,1020" --log=debug --soc_version=Ascend${chip_name}
      ```
      
      运行成功后生成wdsr_bs${bs}.om模型文件。
      
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
      python3 -m ais_bench --model=wdsr_bs${bs}.om  --batchsize=${bs} \
      --input ${prep_data} --output result --output_dirname result_bs${bs} --outfmt BIN
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

   ```
   python3 wdsr_postprocess.py --bin_data_path ./result/result_bs${bs}/ --dataset_path ./data/DIV2K/DIV2K_valid_HR/ --result result_bs${bs}.txt --scale 2
   ```

   参数说明：

   - --参数1：生成推理结果所在路径。
   - --参数2：高分辨率图片所在位置。
   - --参数3：生成结果文件。
   - --参数4：缩放图片大小。

4. 可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

   ```
   python3 -m ais_bench --model=wdsr_bs${bs}.om --loop=500 --batchsize=${bs}
   ```

   参数说明：

   - --model：om模型路径。
   - --loop：纯推理循环次数。
   - --batchsize：批次大小。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，wdsr模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集 | 开源精度                                            | 精度指标（PSNR） |
| ----------- | ---------- | ------ | --------------------------------------------------- | ---------------- |
| Ascend310P3 | 1          | DIV2K  | [34.76](https://github.com/ychfan/wdsr#performance) | 34.7537          |

| 芯片型号    | Batch Size | 性能（FPS）      |
| ----------- | ---------- | ---------------- |
| Ascend310P3 | 1          | 13.09            |
| Ascend310P3 | 4          | 12.20            |
| Ascend310P3 | 8          | 12.04            |
| Ascend310P3 | 16         | 10.03            |
| Ascend310P3 | 32         | 10.05            |
| Ascend310P3 | 64         | 注：超出设备内存 |