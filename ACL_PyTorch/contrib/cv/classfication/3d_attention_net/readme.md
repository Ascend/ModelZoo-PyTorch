# 3D_Attention_Net模型-推理指导

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

`Residual Attention Network` 是一种注意力网络。其受注意力机制和深度神经网络的启发，主要包含数个堆积的Attention Module，每一个Module专注于不同类型的注意力信息，同时提出了 `Attention Residual Learning` 避免了Attention Module简单堆叠引起的负面表达。

- 参考实现：

  ```
  url=https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch.git
  branch=master
  commit_id=44d09fe9afc0d5fba6f3f63b8375069ae9d54a56
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | image    | FLOAT32  | batchsize x 3 x 32 x 32   | NCHW         |

- 输出数据

  | 输出数据 | 大小               | 数据类型 | 数据排布格式 |
  | -------- | --------           | -------- | ------------ |
  | class    | batch_size x class | FLOAT32  | ND           |

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
   cd ACL_PyTorch/contrib/cv/classfication/3d_attention_net              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   git clone https://gitee.com/Ronnie_zheng/MagicONNX.git MagicONNX
   cd MagicONNX && git checkout dev
   pip3 install . && cd ..
   ```

3. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch.git
   cd ResidualAttentionNetwork-pytorch
   git checkout 44d09fe9afc0d5fba6f3f63b8375069ae9d54a56
   cd Residual-Attention-Network
   cp -r model ../..
   cp model_92_sgd.pkl ../..
   cd ../../model/
   patch -p1 <../3d_attention_net.patch
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。

   本模型采用 [CIFAR-10数据集:cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar.html)。解压到 `./data` 目录下（如没有则需创建）。

   数据目录结构请参考：

   ```
   ├── data              
         ├──cifar-10-batches-py  
              │──batched.meta
              │──data_batch_1
              │──data_batch_2
              │── ...
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行预处理脚本，生成数据集预处理后的bin文件:

   ```
   python3 3d_attention_net_preprocess.py --data_path ./data --save_path ./preprocessed_data
   ```

   - 参数说明：

     --data_path: 数据集文件位置。

     --save_path：输出文件位置。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      原仓中包含模型权重文件，位置为 `./ResidualAttentionNetwork-pytorch/Residual-Attention-Network/model_92_sgd.pkl` 。

   2. 导出onnx文件。

      1. 使用脚本导出onnx文件。

         运行3d_attention_net_pkl2onnx.py脚本。

         ```
         # pth转换为ONNX
         mkdir -p models/onnx
         python3 3d_attention_net_pkl2onnx.py --input_file ./ResidualAttentionNetwork-pytorch/Residual-Attention-Network/model_92_sgd.pkl --save_path models/onnx/3d_attention_net.onnx
         ```

         - 参数说明：

           --input_file: 模型权重路径。

           --save_path：导出onnx文件路径。

         获得models/onnx/3d_attention_net.onnx文件。

     2. 优化onnx。

        运行resize_optimize.py脚本优化：

        ```
        python3 resize_optimize.py models/onnx/3d_attention_net.onnx models/onnx/3d_attention_net_resize_optimized.onnx
        ```

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
         mkdir -p models/om
         atc --framework=5 --model=models/onnx/3d_attention_net_resize_optimized.onnx --output=models/om/3d_attention_net_bs1 --input_format=NCHW --input_shape="image:1,3,32,32" --log=debug -soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成模型文件models/om/3d_attention_net_bs1.om。


2. 开始推理验证。

   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
        # 以bs1为例
        mkdir -p results/bs1
        python3 -m ais_bench --model ./models/om/3d_attention_net_bs1.om --input ./preprocessed_data/ --output ./results --output_dirname bs1 --batchsize 1 --outfmt TXT
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

      调用3d_attention_net_postprocess.py脚本与数据集标签比对，获得Accuracy数据。

      ```
      python3 3d_attention_net_postprocess.py --pred_res_path=results/bs1 --data_path=./data --output_path=result_bs1.txt
      ```

      -   参数说明：

        --pred_res_path：推理结果所在路径。
        --data_path：数据集所在路径。
        --output_path：最后结果保存路径。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度：

| device      | Top1 Acc |
|-------------|----------|
| GPU         | 95.4%    |
| Ascend310   | 95.34%   |
| Ascend310P3 | 95.34%   |


性能参考下列数据:

| Input                 | 310性能 | 310P性能 |    T4性能 | 310P/310 | 310P/T4 |
| :-----------------:   |    :--: |    :---: |      :--: |      :-: |     :-: |
| 3d_attention_net_bs1  | 672.122 |  1017.57 |  713.4245 |     1.51 |    1.43 |
| 3d_attention_net_bs4  | 2845.32 |  3572.93 | 2306.4459 |     1.26 |    1.55 |
| 3d_attention_net_bs8  | 3317.80 |  4798.82 | 3224.1293 |     1.45 |    1.49 |
| 3d_attention_net_bs16 | 3533.76 |  7806.96 | 3711.3928 |     2.21 |    2.10 |
| 3d_attention_net_bs32 | 3633.64 |  7742.07 | 3950.9440 |     2.13 |    1.96 |
| 3d_attention_net_bs64 | 3700.00 |  5927.82 | 4217.2961 |     1.60 |    1.41 |
| 最优batch             | 3700.00 |  7806.96 | 4217.2961 |     2.11 |    1.85 |
