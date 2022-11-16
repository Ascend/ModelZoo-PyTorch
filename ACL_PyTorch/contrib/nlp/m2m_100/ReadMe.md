# m2m100模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

m2m100是Meta公司开发的最新seq2seq模型，支持100中语言间的互相翻译，达到多语言翻译SOTA，本模型代码来源于fairseq仓库。



- 参考实现：

  ```
  url=https://github.com/facebookresearch/fairseq.git
  commit_id=
  code_path=https://github.com/facebookresearch/fairseq/tree/main/examples/m2m_100
  model_name=m2m100
  ```
  


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据    | 数据类型 | 大小                    | 数据排布格式 |
  | --------   | --------| -----------------------| ----------- |
  | src_tokens | INT64   |         1 X 90          |     ND      | 


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | ------------------ | ------------ |
  | output   | FLOAT32  | 5 x 1 X 128112       | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.9.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone -b 0.12.2-release https://github.com/facebookresearch/fairseq.git
   ```

2. 安装依赖。

   ```
   参考fairseq官方指导，安装相关依赖

   进入fairseq源码目录

   使用fairseq_m2m100.patch补丁修改代码

   dos2unix fairseq_m2m100.patch
   git apply fairseq_m2m100.patch

   安装fairseq

   python3 setup.py install
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   ```
   以中英互译为例，通过sacrebleu下载数据集

   sacrebleu --echo src -l en-zh -t wmt20 | head -n 32 > raw_input.en-zh.en
   sacrebleu --echo ref -l en-zh -t wmt20 | head -n 32 > raw_input.en-zh.zh

   注：这里head -n 后的32代表下载32条语句作为数据集，用户可以自行修改

   下载数据库字典

   wget https://dl.fbaipublicfiles.com/m2m_100/model_dict.128k.txt
   wget https://dl.fbaipublicfiles.com/m2m_100/language_pairs_small_models.txt 

   下载spm model

   wget https://dl.fbaipublicfiles.com/m2m_100/spm.128k.model
   ```


2. 数据预处理，将原始数据集转换为模型输入的数据。

   顺序执行以下命令完成数据预处理

   ```
   python3 scripts/spm_encode.py --model spm.128k.model --output_format=piece --inputs=raw_input.en-zh.en --outputs=spm.en-zh.en
   python3 scripts/spm_encode.py --model spm.128k.model --output_format=piece --inputs=raw_input.en-zh.zh --outputs=spm.en-zh.zh

   fairseq-preprocess --source-lang en --target-lang zh --testpref spm.en-zh --thresholdsrc 0 --thresholdtgt 0 --destdir data_bin --srcdict data_dict.128k.txt --tgtdict data_dict.128k.txt
   ```



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       下载权重文件
       wget https://dl.fbaipublicfiles.com/m2m_100/418M_last_checkpoint.pt 

   2. 导出onnx文件。

      1. 导出onnx


         ```       
         将权重文件放到fairseq目录下

         将pyacl目录添加到环境变量PYTHONPATH中

         导出onnx的代码已经通过补丁添加到fairseq/sequence_generator.py内，默认为注释状态

         该模型需要分为三部分导出onnx，分别为encoder, decoder_first_step.onnx, decoder.onnx

         encoder.onnx导出代码在828行

         decoder_first_step.onnx导出代码在903行

         decoder.onnx导出代码在959行

         执行命令导出onnx

         fairseq-generate data_bin --batch-size 1 --path 418M_last_checkpoint.pt --fixed-dictionary model_dict.128k.txt -s en -t zh --remove-bpe 'sentencepiece' --beam 5 --task translation_multi_simple_epoch --lang-pairs language_pairs_small_models.txt --decoder-langtok --encoder-langtok src --gen-subset test

         导出onnx分别取消对应的代码注释并执行命令，重复三次后获得三部分onnx。完成后需要将导出用代码恢复注释状态，否则会干扰后续推理。

         ```

         获得m2m_encoder.onnx, m2m_decoder_first_step.onnx, m2m_decoder.onnx文件。


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
         atc --model=m2m_encoder.onnx --framework=5 --output=m2m_encoder --input_format=ND --input_shape="src_tokens:1, 90" --log=info --soc_version=Ascend${chilp_name} --precision_mode=allow_mix_precision --modify_mixlist=ops_info.json

         atc --model=m2m_decoder_first_step.onnx --framework=5 --output=m2m_decoder_first_step --input_format=ND --log=info --soc_version=Ascend${chilp_name} --precision_mode=allow_mix_precision --modify_mixlist=ops_info.json

         因为decoder需要转换为分档模型，档位较多，命令较为复杂，建议直接使用脚本转换
         bash atd_decoder.sh
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --precision_mode: 允许混合精度模式
           -   --modify_mixlist: 采用混合精度的算子列表

           运行成功后生成 m2m_encoder.om, m2m_decoder_first_step.om, m2m_decoder.om

2. 开始推理验证。

   1. 执行推理。
      注释掉导出onnx代码，得到om后，直接执行推理命令即可，注意根据实际路径修改fairseq/sequence_generator.py 781行开始的om模型路径

        ```
          fairseq-generate data_bin --batch-size 1 --path 418M_last_checkpoint.pt --fixed-dictionary model_dict.128k.txt -s en -t zh --remove-bpe 'sentencepiece' --beam 5 --task translation_multi_simple_epoch --lang-pairs language_pairs_small_models.txt --decoder-langtok --encoder-langtok src --gen-subset test
        ```


        推理结果会直接打印，所有数据集推理完后会打印性能和BLEU精度等信息
        精度取决于实际翻译的语句，用户可自行判断打印出的predict句子和target句子意思是否相符



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| 310P3     |        1         |            |            |     35 tokens/s      |