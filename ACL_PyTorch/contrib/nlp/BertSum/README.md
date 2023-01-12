# BertSum模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

BertSum模型主要由句子编码层和摘要判断层组成，其中，`句子编码层` 通过BERT模型获取文档中每个句子的句向量编码，`摘要判断层` 通过三种不同的结构进行选择判断，为每个句子进行打分，最终选取最优的top-n个句子作为文档摘要。

  ```
  url=https://github.com/nlpyang/BertSum
  branch=master
  commit_id=05f8c634197
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | src      | INT64    | batchsize x seq_len       | ND           |
  | segs     | INT64    | batchsize x seq_len       | ND           |
  | clss     | INT64    | batchsize x seq_len       | ND           |
  | mask     | BOOL     | batchsize x seq_len       | ND           |
  | mask_cls | BOOL     | batchsize x seq_len       | ND           |

- 输出数据

  | 输出数据 | 大小               | 数据类型 | 数据排布格式 |
  | -------- | --------           | -------- | ------------ |
  | output   | batch_size x class | FLOAT32  | ND           |
  | mask_cls | batch_size x class | FLOAT32  | ND           |

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
   cd ACL_PyTorch/contrib/nlp/BertSum              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```
   其中`pyrouge`安装较为复杂，完整过程如下（部分依赖如果已安装，则可以跳过）：
   ```
   # 安装Berkeley DB library依赖
   sudo apt-cache search libdb  # 检查当前Berkeley DB library 的版本
   sudo apt-get install libdb5.3-dev  # 安装对应DB版本
   # 安装Perl解释器的DB_File模块
   wget http://www.cpan.org/authors/id/P/PM/PMQS/DB_File-1.835.tar.gz
   tar -zxvf DB_File-1.835.tar.gz
   cd DB_File-1.835
   perl Makefile.PL
   make
   make test # if %%%看到PASS为成功
   sudo make install
   # 安装pyrouge库
   pip3 install pyrouge==0.1.3
   # 下载pyrouge源码
   git clone https://github.com/andersjo/pyrouge.git
   cd pyrouge && git checkout 3b6c415204dbc2c8360a01d92533441f4aae95eb
   pyrouge_set_rouge_path ${work_path}/pyrouge/tools/ROUGE-1.5.5  # 需要为绝对路径, ${work_path}为仓代码路径
   # 需要重新编译WordNet的DB文件
   cd tools/ROUGE-1.5.5/data
   rm WordNet-2.0.exc.db
   cd WordNet-2.0-Exceptions
   cd data/WordNet-2.0-Exceptions/
   ./buildExeptionDB.pl . exc WordNet-2.0.exc.db
   cd ../
   # 软链新生成的文件
   ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
   cd ${work_path}  # 返回原始目录
   ```

2. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/nlpyang/BertSum.git
   cd BertSum && git checkout 05f8c634197
   patch -p1 < bertsum.patch
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。

   本模型采用仓内自带的[预处理数据](https://drive.google.com/open?id=1x0d61LP9UAN389YN00z0Pv-7jQgirVg6)，放到`bert_data`目录下（如不存在，则需要自行创建）。

   数据目录结构请参考：

   ```
   ├──bert_data
    ├──cnndm.test.0.pt
    ├──...
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行“BertSum_preprocess.py”脚本，完成预处理。

   ```
   python BertSum_pth_preprocess.py -bert_data_path ./bert_data/cnndm -out_path ./prep_data
   ```

   - 参数说明：

     -bert_data_path：原始数据集所在路径

     -out_path: 预处理结果所在路径。

  运行成功后，在当前`./prep_data`目录下生成二进制文件夹。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       获取[训练好的模型](https://pan.baidu.com/s/1bM16HNCQHeqbXYhHscmzQA)（提取码：e0nv ）放到当前目录。

   2. 导出onnx文件。

      1. 使用脚本导出onnx文件。

         运行BertSum_pth2onnx.py脚本。

         ```
         python3 BertSum-pth2onnx.py  -bert_data_path ./bert_data/cnndm -onnx_path bertsum_13000_9.onnx -pth_path model_step_13000.pt
         ```

         - 输入参数说明：
           - -bert_data_path: 原始预处理数据路径。
           - -onnx_path: 输出onnx文件路径。
           - -pth_path: 模型权重路径。

         获得bertsum_13000_9.onnx文件。

      2. 优化ONNX文件。

         ```
         # 以bs1为例
         python -m onnxsim ./bertsum_13000_9.onnx ./bertsum_13000_9_sim_bs1.onnx --input-shape "src:1,512" "segs:1,512" "clss:1,37" "mask:1,512" "mask_cls:1,37"
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
         atc --input_format=ND --framework=5 --model=./bertsum_13000_9_sim_bs1.onnx --input_shape="src:1,512;segs:1,512;clss:1,37;mask:1,512;mask_cls:1,37" --output=bertsum_13000_9_sim_bs1 --log=error --soc_version=Ascend310
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成bertsum_13000_9_sim_bs1.om模型文件。



2. 开始推理验证。

   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

        ```
        # 以bs1为例
        python3 -m ais_bench --model "./bertsum_13000_9_sim_bs1.om" --input "./prep_data/pre_data/src,./prep_data/pre_data/segs,./prep_data/pre_data/clss,./prep_data/pre_data/mask,./prep_data/pre_data/mask_cls" --output "./results" --output_dirname bs1 --batchsize 1
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --output_dirname：输出文件名。
             -   --batchsize：模型对应batchsize。


        推理后的输出默认在当前目录results/bs1下。

   3.  精度验证。

      调用BertSum_postprocess.py脚本与数据集标签比对，获得Accuracy数据。

      ```
      mkdir temp  # 创建临时存储文件
      python BertSum_pth_postprocess.py -bert_data_path ./bert_data/cnndm -bert_config_path BertSum/bert_config_uncased_base.json -result_dir ./results/bs1
      ```
      - 输入参数说明：
        - -bert_data_path：原始数据路径。
        - -bert_config_path： 模型配置路径。
        - -result_dir: 推理结果路径。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度参考下列数据:

| device | ROUGE-1 Average_R |
|--------|-------------------|
| 基准   |            42.96% |
| 310    |            42.95% |
| 310P3  |            42.85% |


性能参考下列数据。


| 模型         | 基准性能  | 310性能   | 310P3性能 |
| BertSum bs1  | 61.538fps | 94.281fps | 136.33fps |
| :------:     | :------:  | :------:  | :------:  |
| BertSum bs4  | -         | -         | 133.11fps |
| BertSum bs8  | -         | -         | 138.09fps |
| BertSum bs16 | -         | -         | 138.06fps |
| BertSum bs32 | -         | -         | 137.42fps |
| BertSum bs64 | -         | -         | 118.94fps |
