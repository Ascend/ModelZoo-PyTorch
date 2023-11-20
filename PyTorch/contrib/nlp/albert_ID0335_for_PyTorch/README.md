# Albert for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

Albert是自然语言处理模型，基于Bert模型修改得到。相比于Bert模型，Albert的参数量缩小了10倍，减小了模型大小，加快了训练速度。在相同的训练时间下，Albert模型的精度高于Bert模型。

- 参考实现：

  ```
  url=https://github.com/lonePatient/albert_pytorch 
  commit_id=46de9ec6b54f4901f78cf8c19696a16ad4f04dbc
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/nlp
  ```

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version  | 三方库依赖版本                    |
  | :------------: | :------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  | PyTorch 1.11   | - |
  | PyTorch 2.1   | - |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```



## 准备数据集

1. 获取数据集。

   用户自行下载 `SST-2` 和 `STS-B` 数据集，在模型根目录下创建 `dataset` 目录，并放入数据集。

   数据集目录结构参考如下所示。

   ```
   ├── dataset
         ├──SST-2
              ├──original
              │──dev.tsv
              │──test.tsv
              │──train.tsv   
              |  ...                     
         ├──STS-B  
              ├──original
              │──dev.tsv
              │──test.tsv
              │──train.tsv
              │   ...              
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


## 下载预训练模型
下载 `albert_base_v2` 预训练模型，在模型根目录下创建 `prev_trained_model` 目录，并将预训练模型放置在该目录下。

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=real_data_path         #单卡精度
     bash ./test/train_performance_1p.sh --data_path=real_data_path  #单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=real_data_path         #8卡精度
     bash ./test/train_performance_8p.sh --data_path=real_data_path  #8卡性能 
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path=real_data_path  #8卡评测
     ```
   --data\_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_dir                           //数据集路径
   --model_type                         //模型类型
   --task_name                          //任务名称
   --output_dir                         //输出保存路径
   --do_train                           //是否训练
   --do_eval                            //是否验证
   --num_train_epochs                   //重复训练次数
   --batch-size                         //训练批次大小
   --learning_rate                      //初始学习率
   --fp16                               //是否使用混合精度
   --fp16_opt_level                     //混合精度的level
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V | 0.927 | 517  |   2    |    O1     |      1.5      |
| 8p-竞品V | 0.914 | 3327 |  7   |    O1     |      1.5      |
|  1p-NPU  | 0.932 | 445.21  |   2    |    O2    |      1.8      |
|  8p-NPU  | 0.927 | 3111.56  |  7   |    O2    |      1.8      |


# 版本说明

## 变更

2022.08.24：首次发布。

## FAQ

无。




# Albert模型-推理指导

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

ALBERT是BERT 的“改进版”，主要通过通过Factorized embedding parameterization和Cross-layer parameter sharing两大机制减少参数量，得到一个占用较小的模型，对实际落地有较大的意义，不过由于其主要还是减少参数，不影响推理速度。

  ```
  url=https://github.com/lonePatient/albert_pytorch
  branch=master
  commit_id=46de9ec
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据       | 数据类型 | 大小                      | 数据排布格式 |
  | --------       | -------- | ------------------------- | ------------ |
  | input_ids      | INT64    | batchsize x seq_len       | ND           |
  | attention_mask | INT64    | batchsize x seq_len       | ND           |
  | token_type_ids | INT64    | batchsize x seq_len       | ND           |

  说明：该模型默认的seq_len为128

- 输出数据

  | 输出数据 | 大小               | 数据类型 | 数据排布格式 |
  | -------- | --------           | -------- | ------------ |
  | output   | batch_size x class | FLOAT32  | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                            | 版本    | 环境准备指导                                                                                          |
| ------------------------------------------------------------    | ------- | ------------------------------------------------------------                                          |
| 固件与驱动                                                      | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                            | 6.0.RC1 | -                                                                                                     |
| Python                                                          | 3.7.5   | -                                                                                                     |
| PyTorch                                                         | 1.11.0 | -                                                                                                     |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>
可参考实现
https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/nlp/albert

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master         # 切换到对应分支
   cd ACL_PyTorch/contrib/nlp/albert              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements_for_infer.txt
   git clone https://gitee.com/ascend/msadvisor && cd msadvisor && git checkout master
   cd auto-optimizer && python3 -m pip install .
   cd ../..
   ```

2. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/lonePatient/albert_pytorch.git
   cd albert_pytorch
   git checkout 46de9ec
   patch -p1 < ../albert.patch
   cd ../
   ```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。

   本模型采用[SST-2数据集](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip)，解压到 `albert_pytorch/dataset/SST-2`文件夹下

   数据目录结构请参考：
   ```
   ├──SST-2
    ├──original/
    ├──dev.tsv
    ├──train.tsv
    ├──test.tsv
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。该模型数据预处理需要加载模型，所以需要先下载权重文件：

   获取预训练权重文件，并解压到albert_pytorch/prev_trained_model/albert_base_v2。

   下载[训练好的模型](https://pan.baidu.com/s/1G5QSVnr2c1eZkDBo1W-uRA )（提取码：mehp ）并解压到albert_pytorch/outputs/SST-2。

   执行“Albert_preprocess.py”脚本，完成预处理。

   ```
   python3 Albert_preprocess.py --pth_dir=./albert_pytorch/outputs/SST-2/ --data_dir=./albert_pytorch/dataset/SST-2/ --save_dir ./preprocessed_data_seq128 --max_seq_length 128
   ```
   - 参数说明：

     --pth_dir: 模型权重所在路径

     --data_dir：原始数据集所在路径

     --save_dir: 预处理数据保存路径, 其中gt_label保存在 `${save_dir}/label.npy`
     
     --max_seq_length: 对应的seq长度，默认为128，支持：16/32/64/128


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
   
      数据预处理阶段已经完成模型权重下载。

   2. 导出onnx文件。

      1. 使用脚本导出onnx文件。

         运行“Albert_pth2onnx.py”脚本。

         ```
         # pth转换为ONNX，此处以seq128/bs32为例
         python3 ./Albert_pth2onnx.py --batch_size=32 --pth_dir=./albert_pytorch/outputs/SST-2/ --data_dir=./albert_pytorch/datasets/SST-2/ --onnx_dir=./outputs/ --max_seq_length=128
         ```

         - 参数说明：

           --batch_size: 导出模型batchsize。

           --pth_dir：权重所在路径。
           
           --data_dir: 数据集所在路径。

           --onnx_dir: 输出onnx文件所在目录。
           
           --max_seq_length: 模型对应seq，默认为128，支持：16/32/64/128。

         获得outputs/albert_seq128_bs32.onnx文件。

      2. 优化ONNX文件。

         ```
         # 以seq128/bs32为例
         python3 -m onnxsim ./outputs/albert_seq128_bs32.onnx ./outputs/albert_seq128_bs32_sim.onnx
         python3 opt_onnx.py --input_file ./outputs/albert_seq128_bs32_sim.onnx --output_file ./outputs/albert_seq128_bs32_opt.onnx
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
         #该设备芯片名为Ascend910A （自行替换）
         回显如下：
         +-------------------|-----------------|------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       910A     | OK              | 15.8         42                0    / 0              |
         | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
         +===================+=================+======================================================+
         | 1       910A     | OK              | 15.4         43                0    / 0              |
         | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
         +===================+=================+======================================================+
         ```

      3. 执行ATC命令。
         ```
         # 以seq128/bs32为例
         atc --input_format=ND --framework=5 --model=./outputs/albert_seq128_bs32_opt.onnx --output=./outputs/albert_seq128_bs32 --log=error --soc_version=${chip_name} --input_shape="input_ids:32,128;attention_mask:32,128;token_type_ids:32,128" --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --optypelist_for_implmode：需要指定精度模式的算子。
           -   --op_select_implmode：特定算子需要采取的精度模式。

           运行成功后生成albert_seq128_b32.om模型文件。

         对于`seq16`对应的模型，ATC命令有所区别，如下：
        
         ```
         # 以seq16/bs64为例
         atc --input_format=ND --framework=5 --model=./outputs/albert_seq16_bs64_opt.onnx --output=./outputs/albert_seq16_bs64 --log=error --soc_version=${chip_name} --input_shape="input_ids:64,16;attention_mask:64,16;token_type_ids:64,16" --op_precision_mode=precision.ini
         ```

         - 额外参数说明：

           -   --op_precision_mode：算子精度模式配置输入。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
        # 以bs32为例
        python3 -m ais_bench --model outputs/albert_seq128_bs32.om --input ./preprocessed_data_seq128/input_ids,./preprocessed_data_seq128/attention_mask,./preprocessed_data_seq128/token_type_ids --output results --output_dirname seq128_bs32 --outfmt NPY --batchsize 32
        ```
        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --output_dirname：输出文件名。
             -   --device：NPU设备编号。
             -   --outfmt: 输出数据格式。
             -   --batchsize：推理模型对应的batchsize。


        推理后的输出默认在当前目录outputs/seq128_bs32下。

   3.  精度验证。

      调用Albert_postprocess.py脚本与数据集标签比对，获得Accuracy数据。

      ```
      python3 Albert_postprocess.py --result_dir results/seq128_bs32 --label_path preprocessed_data_seq128/label.npy
      ```

      -   参数说明：

        --result_dir：生成推理结果所在路径。

        --label_path：GT label文件所在路径。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

seq128对应的精度性能如下：

精度:

| device | ACC(seq128) |
|--------|-------------|
| 基准   |       92.8% |
| 910A    |       92.8% |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md   
