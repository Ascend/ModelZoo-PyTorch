# AdvancedEAST模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

  - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

AdvancedEAST是一种用于场景图像文本检测的算法，它主要基于EAST: An Efficient and Accurate Scene Text Detector，并进行了重大改进，使长文本预测更加准确。


- 参考论文：[Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou, Weiran He, Jiajun Liang. EAST: An Efficient and Accurate Scene Text Detector. (2017)](https://arxiv.org/abs/1704.03155v2)

- 参考实现：

  ```
  url=https://github.com/BaoWentz/AdvancedEAST-PyTorch 
  branch=master 
  commit_id=a835c8cedce4ada1bc9580754245183d9f4aaa17
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 736 x 736 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | output1  | FLOAT32  | batchsize x 7 x 184 x 184 | ND           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套       | 版本                               | 环境准备指导                                                 |
  | ---------- | ---------------------------------- | ------------------------------------------------------------ |
  | 固件与驱动 | 1.0.16（NPU驱动固件版本为5.1.RC2） | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN       | 5.1.RC2                            | -                                                            |
  | Python     | 3.7.5                              | -                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/BaoWentz/AdvancedEAST-PyTorch -b master 
   cd AdvancedEAST-PyTorch 
   git reset a835c8cedce4ada1bc9580754245183d9f4aaa17 --hard 
   cd ..
   ```

2. 安装依赖。

   ```
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持天池ICPR数据集中的1000张图片作为验证集。下载链接: https://pan.baidu.com/s/1NSyc-cHKV3IwDo6qojIrKA，密码: ye9y。下载ICPR_text_train_part2_20180313.zip和[update] ICPR_text_train_part1_20180316.zip两个压缩包，新建目录icpr和子目录icpr/image_10000、icpr/txt_10000，将压缩包中image_9000、image_1000中的图片文件解压至image_10000中，将压缩包中txt_9000、txt_1000中的标签文件解压至txt_10000中。目录结构如下：

   ```
   icpr
   ├── image_10000           //验证集图片  
        ├── img1.jpg
   └── txt_10000             // 标签文件夹
        ├── img1.txt
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行AdvancedEAST_preprocess.py脚本，完成预处理。

   ```
   python AdvancedEAST_preprocess.py icpr prep_dataset
   ```

    -   参数说明：
         - icpr：数据集的路径。
         - prep_dataset：预处理之后bin文件存放的文件夹。

   运行成功生成文件夹prep_dataset，存放着预处理之后的二进制文件。




## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件[3T736_best_mF1_score.pth](https://pan.baidu.com/s/1ZGag4Ar7Yf5P_rmBrLhcWw?pwd=ttnb)。

   2. 导出onnx文件。

      使用AdvancedEAST_pth2onnx.py导出onnx文件。

      运行AdvancedEAST_pth2onnx.py脚本。

      ```
      python AdvancedEAST_pth2onnx.py 3T736_best_mF1_score.pth AdvancedEAST_dybs.onnx
      ```

      - 参数说明：
        - 3T736_best_mF1_score.pth：pth权重文件路径。
        - AdvancedEAST_dybs.onnx：导出onnx的文件路径。

      获得AdvancedEAST_dybs.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

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

      3. 执行ATC命令。

         ```
         atc --framework=5 --model=AdvancedEAST_dybs.onnx --output=AdvancedEAST_bs1 --input_format=NCHW --input_shape='input_1:1,3,736,736' --log=error --soc_version=Ascend${chip_name} --auto_tune_mode='RL,GA'
         ```

         - 参数说明：
           - --model：为ONNX模型文件。
           - --framework：5代表ONNX模型。
           - --output：输出的OM模型。
           - --input_format：输入数据的格式。
           - --input_shape：输入数据的shape。
           - --log：日志级别。
           - --soc_version：处理器型号。
           - --auto_tune_mode: 设置算子的自动调优模式。
           
          运行成功后生成<u>***AdvancedEAST_bs1.om***</u>模型文件。

2. 开始推理验证。

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)。

   2. 执行推理。

      ```
      python -m ais_bench --model AdvancedEAST_bs1.om  --input prep_dataset/ --output ./result --output_dir dumpout_bs1 --batchsize 1
      ```

      -   参数说明：

           -   --model：模型类型。
           -   --input：om文件路径。
           -   --output：输出文件目录。
           -   --output_dir：输出文件子目录。
           -   --batchsize：数据集的batchsize。

      推理后的输出默认在当前目录result下。

      >**说明：** 
      >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见 [参数说明](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench#参数说明)。

   3. 精度验证。

      调用脚本AdvancedEAST_postprocess.py，获得精度数据。

      ```
      python AdvancedEAST_postprocess.py icpr result/dumpout_bs1
      ```

      - 参数说明：

        - icpr：数据集路径
        - result/dumpout_bs1：推理结果所在路径。

      > 注意：后处理需要用到libgeos_c.so，若报错请安装系统对应的包，如Ubuntu执行以下命令：sudo apt-get install libgeos-dev。

   4. 性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
      python -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型的路径。
        - --batchsize：推理的batchsize。
        - --loop：推理循环的次数。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号    | Batch Size | 数据集 | 精度            | 性能     |
| ----------- | ---------- | ------ | --------------- | -------- |
| Ascend310P3 |       1       |   ICPR     | f1-score:52.08% | 137.9101 |
| Ascend310P3 |       4       |   ICPR     | f1-score:52.08% | 133.0594 |
| Ascend310P3 |       8       |   ICPR     | f1-score:52.08% | 131.4062 |
| Ascend310P3 |       16      |   ICPR     | f1-score:52.08% | 131.5172 |
| Ascend310P3 |       32      |   ICPR     | f1-score:52.08% | 131.3700 |
| Ascend310P3 |       64      |    ICPR    | f1-score:52.08% | 131.7403 |

