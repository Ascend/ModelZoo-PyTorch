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

  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
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



1. 安装依赖。

   ```
   pip3 install -r requirment.txt
   ```
2. 下载开源代码仓和gitee实现，并将对应的gitee中的代码复制到开源代码仓下；最终结构如下：

   ```
   ├── big_tansfer
      ├── bit_preprocess.py
         ├── bit_postprocess.py  
         ├── bit_pth2onnx.py 
         ├── LICENSE  
      ├── README.md 
         ├── requirements.txt
         ...
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   此模型CIFAR10。下载好数据集后（cifar-10-python.tar.gz），在big_transfer目录下创建文件夹，命名为DATADIR，然后将数据集放置在DATADIR文件夹内并解压。这一步可使用步骤2的脚本实现。

   下载链接：[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

   目录结构如下：

    ```
   ├── big_transfer
   	├── DATADIR
   		├── cifar-10-batches-py
   			├──test_batch
   			├──batches.meta
   			├──readme.html
   			├──data_batch_x
   	├── cifar-10-python.tar.gz
    ```

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   对CIFAR10中的图片进行裁剪遮挡处理，并将结果放置在big_transfer/dataset_bin目录下并生成标签文件label.txt。预处理后的数据输出格式为bin。

   ```
   python3 bit_preprocess.py --dataset_path DATADIR --save_path dataset_bin
   ```
   参数说明：
   •	dataset_path: 数据路径
   •	save_path: 保存路径



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [bit.pth](https://pan.baidu.com/s/1WHVpYbKQVTNYJupsJs8FWg?pwd=3jnx), access code "3jnx"

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

      3. 执行ATC命令（以 bs=16 为例）。

         ```
         atc --framework=5 --model=./bit_bs16_sim.onnx --input_format=NCHW --input_shape="image:16,3,128,128" --output=bit_bs16 --log=debug --soc_version=Ascend${chip_name}
         
         # 备注：Ascend${chip_name}请根据实际查询结果填写
         ```

         参数说明：
         -   --model：为ONNX模型文件。
         -   --framework：5代表ONNX模型。
         -   --output：输出的OM模型。
         -   --input\_format：输入数据的格式。
         -   --input\_shape：输入数据的shape。
         -   --log：日志级别。
         -   --soc\_version：处理器型号。
         -   --insert\_op\_conf:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。

           

2. 开始推理验证。

   a. 使用ais_bench工具进行推理。

      参考[ais_bench工具源码地址](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)安装将工具编译后的压缩包放置在当前目录；解压工具包，安装工具压缩包中的whl文件；

      ```
      pip3 install ./aclruntime-{version}-cp37-cp37m-linux_xxx.whl
      pip3 install ./ais_bench-{version}-py3-none-any.whl
      ```

   b. 执行推理。

      ```
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      python3 -m ais_bench --model ./bit_bs16.om --input ./dataset_bin/ --output ./result/ --outfmt BIN --batchsize 16
      ```

      参数说明：   
      - --model：模型地址
      - --input：预处理完的数据集文件夹
      - --output：推理结果保存地址
      - --outfmt：推理结果保存格式
   
      **说明：** 执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见 --help命令。
   
   c. 精度验证。

      统计推理输出的Top 1-5 Accuracy
      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据。
   
      ```
      python3 bit_postprocess.py --output_dir ${result_dir} --label_path ${gt_file}
      ```

      参数说明：
      - --output_dir：为推理结果所在路径
      - --label_path：为标签数据文件所在路径

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
