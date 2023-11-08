# Bert_Chinese for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder，是一种用于自然语言处理（NLP）的预训练技术。Bert-base模型是一个12层，768维，12个自注意头（self attention head）,110M参数的神经网络结构，它的整体框架是由多层transformer的编码器堆叠而成的。

- 参考实现：

  ```
  url=https://github.com/huggingface/transformers
  commit_id=d1d3ac94033b6ea1702b203dcd74beab68d42d83
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/nlp
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                      |
  | :--------: | :----------------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  | PyTorch 1.11   | - |
  | PyTorch 2.1   | - |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖：

  ```
  pip install -r requirements.txt
  ```

- 安装transformers：

  ```
  cd transformers
  pip3 install -e ./
  cd ..
  ```

## 准备数据集

1. 获取数据集。

    下载 `zhwiki` 数据集。

    解压得到zhwiki-latest-pages-articles.xml。

    ```
    bzip2 -dk zhwiki-latest-pages-articles.xml.bz2
    ```

    使用模型根目录下的WikiExtractor.py提取文本，其中extracted/wiki_zh为保存路径，不要修改。

    ```
    python3 WikiExtractor.py zhwiki-latest-pages-articles.xml -b 100M -o extracted/wiki_zh
    ```

    将多个文档整合为一个txt文件，在模型根目录下执行。

    ```
    python3 WikicorpusTextFormatting.py --extracted_files_path extracted/wiki_zh --output_file zhwiki-latest-pages-articles.txt
    ```

    最终生成的文件名为zhwiki-latest-pages-articles.txt。

    Bert-base下载配置模型和分词文件。

    ```
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/bert-base-chinese
    ```

    将下载下的bert-base-chinese放置在模型根目录下。

    Bert-large下载配置模型和分词文件。

    ```
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/algolet/bert-large-chinese
    ```

    将下载下的bert-large-chinese放置在模型根目录下。


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练、单机8卡训练以及双机多卡训练。

   - 单机单卡训练

     启动base单卡训练。

     ```
     bash test/train_full_1p.sh --data_path=dataset_file_path --batch_size=32 --model_size=base --device_id=0  # 单卡精度训练
     bash test/train_performance_1p.sh --data_path=dataset_file_path --batch_size=32 --model_size=base    # 单卡性能训练   
     ```
     启动large单卡训练。

     ```
     bash test/train_full_1p.sh --data_path=dataset_file_path --batch_size=16 --model_size=large --device_id=0 --warmup_ratio=0.1 --weight_decay=0.00001  # 单卡精度训练
     bash test/train_performance_1p.sh --data_path=dataset_file_path --batch_size=16 --model_size=large --warmup_ratio=0.1 --weight_decay=0.00001    # 单卡性能训练   
     ```

   - 单机8卡训练

     启动base8卡训练。

     ```
     bash test/train_full_8p.sh --data_path=dataset_file_path --batch_size=32 --model_size=base     # 8卡精度训练
     bash test/train_performance_8p.sh --data_path=dataset_file_path --batch_size=32 --model_size=base    # 8卡性能训练  
     ```
     启动large8卡训练。

     ```
     bash test/train_full_8p.sh --data_path=dataset_file_path --batch_size=16 --model_size=large --warmup_ratio=0.1 --weight_decay=0.00001   # 8卡精度训练
     bash test/train_performance_8p.sh --data_path=dataset_file_path --batch_size=16 --model_size=large --warmup_ratio=0.1 --weight_decay=0.00001   # 8卡性能训练  
     ```

   - 多机多卡训练
   
     启动base多机多卡训练。

     ```
     bash test/train_full_multinodes.sh --data_path=dataset_file_path --batch_size=32 --model_size=base --nnodes=node_number --node_rank=node_id --master_addr=x.x.x.x --master_port=xxxx  # 多机多卡精度训练
     bash test/train_performance_multinodes.sh --data_path=dataset_file_path --batch_size=32 --model_size=base --nnodes=node_number --node_rank=node_id --master_addr=x.x.x.x --master_port=xxxx  #多机多卡性能训练
     ```
     
     启动large多机多卡训练。

     ```
     bash test/train_full_multinodes.sh --data_path=dataset_file_path --batch_size=16 --model_size=large --nnodes=node_number --node_rank=node_id --master_addr=x.x.x.x --master_port=xxxx --warmup_ratio=0.1 --weight_decay=0.00001 # 多机多卡精度训练
     bash test/train_performance_multinodes --data_path=dataset_file_path --batch_size=16 --model_size=large --nnodes=node_number --node_rank=node_id --master_addr=x.x.x.x --master_port=xxxx --warmup_ratio=0.1 --weight_decay=0.00001 # 多机多卡性能训练
     ```

     ```
       --data_path：  数据集路径
       --device_number: 每台服务器上要使用的训练卡数
       --model_size： 训练model是base或者是large
       --device_id：  单卡训练时所使用的device_id
       --node_rank:   集群节点序号，master节点是0， 其余节点依次加1
       --master_addr：master节点服务器的ip
       --master_port: 分布式训练中,master节点使用的端口
     ```
   
   - 双机8卡训练  
     启动双机8卡训练。

     ```
     bash ./test/train_cluster_8p.sh --data_path=real_data_path --node_rank=node_id --master_addr=x.x.x.x --master_port=xxxx 
     ```
     
     ```
     --node_rank                              //集群节点序号，master节点是0，其余节点依次加1
     --master_addr                            //master节点服务器的ip
     --master_port                            //分布式训练中，master节点使用的端口
     --data_path                              //数据集路径,需写到数据集的一级目录。
     ```
   
   模型训练脚本参数说明如下。

    ```
    公共参数：
    --config_name                            //模型配置文件
    --model_type                             //模型类型
    --tokenizer_name                         //分词文件路径
    --train_file                             //数据集路径
    --eval_metric_path                       //精度评估处理脚本路径
    --line_by_line                           //是否将数据中一行视为一句话
    --pad_to_max_length                      //是否对数据做padding处理
    --remove_unused_columns                  //是否移除不可用的字段
    --save_steps                             //保存的step间隔
    --overwrite_output_dir                   //是否进行覆盖输出
    --per_device_train_batch_size            //每个卡的train的batch_size
    --per_device_eval_batch_size             //每个卡的evaluate的batch_size
    --do_train                               //是否进行train
    --do_eval                                //是否进行evaluate
    --fp16                                   //是否使用混合精度
    --fp16_opt_level                         //混合精度level
    --loss_scale                             //loss scale值
    --use_combine_grad                       //是否开启tensor叠加优化
    --optim                                  //优化器
    --output_dir                             //输出保存路径
    ```
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

3. 在线推理  
   - 启动在线推理。
 
    ```
    bash ./test/train_eval_1p.sh --data_path=real_data_path --device_id=xxx --checkpoint=real_checkpoint_path
    ```

    ```
    --data_path： 数据集路径
    --device_id：  在线推理时所使用的device_id
    --checkpoint:  权重文件目录
    ```

# 训练结果展示

**表2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |  - | - |  3   |    O2     |      1.5      |
| 8p-竞品V |  0.59 | 898 |  3   |    O2     |      1.5      |
| 1p-NPU  |  - | 128.603  |  3   |    O2    |      1.8      |
| 8p-NPU  |  0.59 | 936.505  |  3   |    O2    |      1.8      |


# 版本说明

## 变更

2022.08.24：首次发布

## FAQ

1. Q:第一次运行报类似"xxx **socket timeout** xxx"的错误该怎么办？

   A:第一次运行tokenizer会对单词进行预处理，根据您的数据集大小，耗时不同，若时间过长，可能导致等待超时。此时可以通过设置较大的超时时间阈值尝试解决：

    （1）设置pytorch框架内置超时时间，修改脚本中的distributed_process_group_timeout（单位秒）为更大的值，例如设置为7200：
   
    ```
    --distributed_process_group_timeout 7200
    ```

    （2）设置HCCL的建链时间为更大的值，修改env.sh中环境变量HCCL_CONNECT_TIMEOUT（单位秒）的值：

    ```
    export HCCL_CONNECT_TIMEOUT=7200
    ```
2. Q:如果训练报wandb.error.UsageError:api_key not configured (no-tty)的错误该怎么办?
  
   A:export WANDB_DISABLED=1



# Bert_Base_Chinese模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

- [其他下游任务](#ZH-CN_TOPIC_0000001126121892)


## 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

`BERT`来自 Google 的论文`Pre-training of Deep Bidirectional Transformers for Language Understanding`，`BERT` 是`Bidirectional Encoder Representations from Transformers`的首字母缩写，整体是一个自编码语言模型。`Bert_Base_Chinese`是`BERT`模型在中文语料上训练得到的模型。

  ```shell
  url=https://huggingface.co/bert-base-chinese
  commit_id=38fda776740d17609554e879e3ac7b9837bdb5ee
  mode_name=Bert_Base_Chinese
  ```

### 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据       | 数据类型 | 大小                      | 数据排布格式 |
  | --------       | -------- | ------------------------- | ------------ |
  | input_ids      | INT64    | batchsize x seq_len       | ND           |
  | attention_mask | INT64    | batchsize x seq_len       | ND           |
  | token_type_ids | INT64    | batchsize x seq_len       | ND           |

- 输出数据

  | 输出数据 | 大小               | 数据类型 | 数据排布格式 |
  | -------- | --------           | -------- | ------------ |
  | output   | batch_size x class | FLOAT32  | ND           |


## 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                            | 版本    | 环境准备指导                                                                                          |
| ------------------------------------------------------------    | ------- | ------------------------------------------------------------                                          |
| 固件与驱动                                                      | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                            | 6.0.RC1 | -                                                                                                     |
| Python                                                          | 3.7.5   | -                                                                                                     |
| PyTorch                                                         | 1.5.0+ | -                                                                                                     |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

## 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>
可参考实现https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/nlp/Bert_Base_Chinese_for_Pytorch

### 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master         # 切换到对应分支
   cd ACL_PyTorch/built-in/nlp/Bert_Base_Chinese_for_Pytorch              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements_for_infer.txt
   git clone https://gitee.com/Ronnie_zheng/MagicONNX.git MagicONNX
   cd MagicONNX && git checkout dev
   pip3 install . && cd ..
   ```

2. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```shell
   GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/bert-base-chinese
   ```

### 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。

   如果你想重新处理zhwiki的原始数据，可按照以下步骤操作。

   下载zhwiki原始数据：

   ```
   wget https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2 --no-check-certificate
   ```

   解压得到zhwiki-latest-pages-articles.xml

   ```
   bzip2 -dk zhwiki-latest-pages-articles.xml.bz2
   ```

   下载预处理脚本：

   ```shell
    wget https://github.com/natasha/corus/raw/master/corus/third/WikiExtractor.py
   ```

   使用WikiExtractor.py提取文本，其中extracted/wiki_zh为保存路径，建议不要修改：

   ```
   python3 WikiExtractor.py zhwiki-latest-pages-articles.xml -b 100M -o extracted/wiki_zh
   ```

   将多个文档整合为一个txt文件，在本工程根目录下执行

   ```
   python3 WikicorpusTextFormatting.py --extracted_files_path extracted/wiki_zh --output_file zhwiki-latest-pages-articles.txt
   ```

   最终生成的文件名为zhwiki-latest-pages-articles.txt (也可直接采用处理好的文件)

   从中分离出验证集：

   ```shell
   python3 split_dataset.py zhwiki-latest-pages-articles.txt zhwiki-latest-pages-articles_validation.txt
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行preprocess.py脚本，完成预处理。

   ```shell
   # 输入参数：${input_patgh} ${model_dir} ${save_dir} ${seq_length}
   python3 preprocess.py ./zhwiki-latest-pages-articles_validation.txt ./bert-base-chinese ./input_data/ 384
   ```

   - 参数说明：第一个参数为zhwiki数据集分割得到验证集文件路径，第二个参数为源码路径（包含模型配置文件等），第三个参数为输出预处理数据路径，第四个参数为sequence长度。

### 模型推理<a name="section741711594517"></a>

1. 模型转换

   1. 获取权重文件

      获取权重文件并转换成cpu适配权重替换`bert-base-chinese`目录下的文件：

      ```shell
      mv pytorch_model.bin bert-base-chinese
      ```

   2. 导出onnx文件

      ```shell
      # 输入参数：${model_dir} ${output_path} ${seq_length} 
      python3 pth2onnx.py ./bert-base-chinese ./bert_base_chinese.onnx 384
      ```
      
      - 输入参数说明：第一个参数为源码仓路径（包含配置文件等），第二个参数为输出onnx文件路径，第三个参数为sequence长度。

   3. 优化onnx文件

      ```shell
      # 修改优化模型：${bs}:[1, 4, 8, 16, 32, 64],${seq_len}:384
      python3 -m onnxsim ./bert_base_chinese.onnx ./bert_base_chinese_bs${bs}.onnx --input-shape "input_ids:${bs},${seq_len}" "attention_mask:${bs},${seq_len}" "token_type_ids:${bs},${seq_len}"
      python3 fix_onnx.py bert_base_chinese_bs${bs}.onnx bert_base_chinese_bs${bs}_fix.onnx
      ```

   4. 使用ATC工具将ONNX模型转OM模型。

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
         ```shell
         # bs:[1, 4, 8, 16, 32, 64]
         atc --model=./bert_base_chinese_bs${bs}_fix.onnx --framework=5 --output=./bert_base_chinese_bs${bs} --input_format=ND --log=debug --soc_version=${chip_name} --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成bert_base_chinese_bs${bs}.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
        # 以bs1模型推理为例
        mkdir -p ./output_data/bs1
        python3 -m ais_bench --model ./bert_base_chinese_bs1.om --input ./input_data/input_ids,./input_data/attention_mask,./input_data/token_type_ids --output ./output_data/ --output_dirname bs1 --batchsize 1 --device 1
        ```
        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --output_dirname：输出文件名。
             -   --device：NPU设备编号。
             -   --outfmt: 输出数据格式。
             -   --batchsize：推理模型对应的batchsize。


        推理后的输出默认在当前目录outputs/bs1下。

   3.  精度验证。

      调用postprocess.py脚本与数据集标签比对，获得Accuracy数据。

      ```
      # 以bs1模型推理为例
      # 输入参数：${result_dir} ${gt_dir} ${seq_length}
      python3 postprocess.py ./output_data/bs1 ./input_data/labels 384
      ```
      
      - 参数说明：第一个参数为推理结果路径，第二个参数为gt labe所在路径，第三个参数为sequence长度。

## 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度：

|       模型        | NPU离线推理精度 |
| :---------------: | :-------------: |
| Bert-Base-Chinese |   Acc: 59.07%   |


## 其他下游任务<a name="ZH-CN_TOPIC_0000001126121892"></a>

+ [序列标注(Sequence Labeling)](downstream_tasks/sequence_labeling/README.md)


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
