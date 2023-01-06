# Big_transfer-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  ******



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

在训练视觉的深度神经网络时，预训练表征的转移提高了采样频率并简化了超参数的调整。该模型重新审视了在大型数据集上进行预训练并在目标任务上对模型进行微调的范式，称之为Big Transfer(BiT)。该模型在多个数据集上实现理强大的效果，在CIFAR-10上达到了97.6%，在其他数据集上也表现出不错的效果。

- 参考论文：Kolesnikov A, Beyer L, Zhai X, et al. Big transfer (bit): General visual representation learning[C]//European conference on computer vision. Springer, Cham, 2020: 491-507. [论文链接](https://arxiv.org/abs/1912.11370)


- 参考实现：

  ```
  url= https://github.com/google-research/big_transfer
  branch=master 
  commit_id=140de6e704fd8d61f3e5ea20ffde130b7d5fd065
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 128 x 128 | NCHW         |


- 输出数据

  | 输出数据 | 大小 | 数据类型 | 数据排布格式 |
  | -------- | ---- | -------- | ------------ |
  | output1  | 1    | FLOAT32  | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.17 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | >1.8.0  | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

1. 下载开源代码仓

   ```
   git clone https://github.com/google-research/big_transfer.git
   cd big_transfer
   git reset --hard 140de6e704fd8d61f3e5ea20ffde130b7d5fd065
   ```

2. 将本仓代码复制到开源代码仓下。

3. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   此模型CIFAR10。下载好数据集后（[cifar-10-python.tar.gz](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)），上传到服务器任意路径${data_dir}并解压。目录结构如下：

    ```
   ├── ${data_dir}
   	├── cifar-10-batches-py
   		├──test_batch
   		├──batches.meta
   		├──readme.html
   		├──data_batch_x
    ```
   
2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   对CIFAR10中的图片进行裁剪遮挡处理，并将结果放置在big_transfer/dataset_bin目录下并生成标签文件label.txt。预处理后的数据输出格式为bin。

   ```
   python3 bit_preprocess.py --dataset_path ${data_dir} --save_path ${save_dir} --label_path ${gt_file}
   ```
   参数说明：
   
   - --dataset_path: 数据路径
   - --save_path: 保存路径
   - --label_path：生成真值标签文件

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      下载[bit.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Big-transfer/PTH/bit.pth)权重文件，并放置在工作目录`big_transfer`下

   2. 导出onnx文件。

      1. 运行bit_pth2onnx.py脚本文件实现模型转换。

         ```
         python3 bit_pth2onnx.py bit.pth bit.onnx
         ```

         获得bit.onnx文件。
      
      2. 运行如下脚本简化模型。

         ```
         python3 -m onnxsim bit.onnx bit_bs${bs}_sim.onnx --input-shape ${bs},3,128,128
         ```

         其中`bs`为批次大小，获得简化后的onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （请根据实际芯片填入）
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
         # bs=[1,4,8,16,32,64]
         atc --framework=5 --model=./bit_bs${bs}_sim.onnx --input_format=NCHW --input_shape="image:${bs},3,128,128" --output=bit_bs${bs} --log=error --soc_version=Ascend${chip_name}
         ```
         
         参数说明：
         -   --model：为ONNX模型文件。
         -   --framework：5代表ONNX模型。
         -   --output：输出的OM模型。
         -   --input\_format：输入数据的格式。
         -   --input\_shape：输入数据的shape。
         -   --log：日志级别。
         -   --soc\_version：处理器型号。  
         
   
2. 开始推理验证。

   a. 安装ais_bench推理工具。

   参考[ais-bench工具源码地址](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)安装将工具编译后的压缩包放置在当前目录；解压工具包，安装工具压缩包中的whl文件；

   b. 执行推理。
   
      ```
      python3 -m ais_bench --model ./bit_bs${bs}.om --input ${save_dir} --output result --output_dirname result_bs${bs} --outfmt BIN --batchsize ${bs}
      ```

      参数说明：   
      - --model：模型地址
      - --input：预处理完的数据集文件夹
      - --output：推理结果保存地址
      - --output_dirname：推理结果子文件夹
      - --outfmt：推理结果保存格式
      - --batchsize：批次大小
   
   c. 精度验证。
   
   统计推理输出的Top 1-5 Accuracy，调用脚本与真值标签比对，可以获得精度数据。
   
      ```
      python3 bit_postprocess.py --output_dir ${result_dir} --label_path ${gt_file}
      ```
   
      参数说明：
      - --output_dir：为推理结果所在路径，这里为result/result_bs${bs}
      - --label_path：为标签数据文件所在路径
   
   d. 可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

   ```
   python3 -m ais_bench --model=bit_bs${bs}.om --loop=50 --batchsize=${bs}
   ```
   
   参数说明：
   
   - --model：om模型路径。
   - --batchsize：批次大小。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，精度与性能参考下列数据。

| 芯片型号 | Batch Size | 数据集   | 精度          | 性能    |
| -------- | ---------- | -------- | ------------- | ------- |
| 310P3    | 1          | CIFIR-10 | top-1: 97.62% | 537.29 |
| 310P3    | 4         | CIFIR-10 | top-1: 97.62% | 1388.11 |
| 310P3    | 8         | CIFIR-10 | top-1: 97.62% | 1707.27 |
| 310P3    | 16         | CIFIR-10 | top-1: 97.62% | 1758.00 |
| 310P3    | 32         | CIFIR-10 | top-1: 97.62% | 1655.13 |
| 310P3    | 64         | CIFIR-10 | top-1: 97.62% | 1619.81 |
