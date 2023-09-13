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
  commit_id=4a388e64cd646ed7d7ad8de8fae55df2b8eea91d
  code_path=https://github.com/facebookresearch/fairseq/tree/main/examples/m2m_100
  model_name=m2m100
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | INT64 | 1 x 90        | ND |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output   | FLOAT32  | 5 x 1 X 128112      | ND |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.9.0  | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。**<u>*此处获取指获取第三方开源代码仓的命令*</u>**

   ```
   git clone -b 0.12.2-release https://github.com/facebookresearch/fairseq.git
   cd fairseq
   git apply ../fairseq_m2m100.patch
   ```

2. 安装依赖。

   ```
   pip3 install torch==1.9.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
   pip3 install -r ../requirements.txt
   python3 setup.py install
   ```

3. 安装ais-infer推理工具。

   ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）


   注：工作目录均为`m2m_200/fairseq`。

   a.推理使用英译中数据集，通过sacrebleu下载数据集。
   
    ```
    python3 -m sacrebleu --echo src -l en-zh -t wmt20 | head -n 200 > raw_input_200.en-zh.en
    python3 -m sacrebleu --echo ref -l en-zh -t wmt20 | head -n 200 > raw_input_200.en-zh.zh
    ```
   
   b.筛选出32条数据用于离线推理。
   
   ```
   python3 ../sort_data.py
   ```
   
   在当前工作目录下生成sort_input_32.en-zh.en、sort_input_32.en-zh.zh文件。

   c.下载数据库字典
	
    ```
    wget https://dl.fbaipublicfiles.com/m2m_100/model_dict.128k.txt
    wget https://dl.fbaipublicfiles.com/m2m_100/language_pairs_small_models.txt
    ```

   d.获取模型`spm.128k.model`，权重文件`418M_last_checkpoint.pt`。
   
    ```
    wget https://dl.fbaipublicfiles.com/m2m_100/spm.128k.model
    wget https://dl.fbaipublicfiles.com/m2m_100/418M_last_checkpoint.pt
    ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

    数据预处理命令为：
   
    ```
    python3 scripts/spm_encode.py --model spm.128k.model --output_format=piece --inputs=sort_input_32.en-zh.en --outputs=spm.en-zh.en
    python3 scripts/spm_encode.py --model spm.128k.model --output_format=piece --inputs=sort_input_32.en-zh.zh --outputs=spm.en-zh.zh

    fairseq-preprocess --source-lang en --target-lang zh --testpref spm.en-zh --thresholdsrc 0 --thresholdtgt 0 --destdir data_bin --srcdict model_dict.128k.txt --tgtdict model_dict.128k.txt
    ```

    参数说明：spm_encode.py脚本参数说明通过`python3 scripts/spm_encode.py -h`命令查看，fairseq-preprocess工具参数说明通过`fairseq-preprocess -h`命令查看。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
   
   模型使用开源仓提供的预训练好的权重：418M_last_checkpoint.pt，[下载链接](https://dl.fbaipublicfiles.com/m2m_100/418M_last_checkpoint.pt)，下载完成后将权重文件存放于`m2m_200/fairseq`工作目录下。

   - 注：数据处理步骤已获取`418M_last_checkpoint.pt`。
    
   2. 导出onnx文件。

      1. 转onnx代码以通过补丁方式打包在fairseq源码中。
	     
         m2m_100模型的推理过程分为Encoder、DecoderFirstStep、Decoder三个步骤，因此需要导出三个onnx模型，命令如下：
		 
         ```
         export EXPORT_ONNX_MODE=encoder
         fairseq-generate data_bin \
            --batch-size 1 \
            --path 418M_last_checkpoint.pt \
            --fixed-dictionary model_dict.128k.txt \
            -s en \
            -t zh \
            --remove-bpe 'sentencepiece' \
            --beam 5 \
            --task translation_multi_simple_epoch \
            --lang-pairs language_pairs_small_models.txt \
            --decoder-langtok \
            --encoder-langtok src \
            --gen-subset test

         export EXPORT_ONNX_MODE=first_step_decoder
         fairseq-generate data_bin \
            --batch-size 1 \
            --path 418M_last_checkpoint.pt \
            --fixed-dictionary model_dict.128k.txt \
            -s en \
            -t zh \
            --remove-bpe 'sentencepiece' \
            --beam 5 \
            --task translation_multi_simple_epoch \
            --lang-pairs language_pairs_small_models.txt \
            --decoder-langtok \
            --encoder-langtok src \
            --gen-subset test

         export EXPORT_ONNX_MODE=decoder
         fairseq-generate data_bin \
            --batch-size 1 \
            --path 418M_last_checkpoint.pt \
            --fixed-dictionary model_dict.128k.txt \
            -s en \
            -t zh \
            --remove-bpe 'sentencepiece' \
            --beam 5 \
            --task translation_multi_simple_epoch \
            --lang-pairs language_pairs_small_models.txt \
            --decoder-langtok \
            --encoder-langtok src \
            --gen-subset test
         unset EXPORT_ONNX_MODE
         ```

        参数说明：fairseq-generate参数说明通过`fairseq-generate -h`命令查看。

        命令执行完成后生成m2m_encoder.onnx、m2m_decoder_first_step.onnx、m2m_decoder.onnx三个onnx模型文件。

        - 注：转出onnx模型后一定要执行`unset EXPORT_ONNX_MODE`命令取消EXPORT_ONNX_MODE环境变量，否则将后影响后面推理。


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

         ATC命令参数说明通过`atc -h`命令参看。

         a.将m2m_encoder.onnx转为m2m_encoder.om。

         ```
         atc --model=m2m_encoder.onnx \
            --framework=5 \
            --output=m2m_encoder \
            --input_format=ND \
            --input_shape="src_tokens:1, 90" \
            --log=error \
            --soc_version=Ascend${chip_name} \
            --precision_mode=allow_mix_precision \
            --modify_mixlist=../ops_info.json
         ```
         
         b.将m2m_decoder_first_step.onnx转为m2m_decoder_first_step.om。
         
         ```
         atc --model=m2m_decoder_first_step.onnx \
            --framework=5 \
            --output=m2m_decoder_first_step \
            --input_format=ND \
            --log=error \
            --input_shape="prev_output_tokens:5,1;input.13:90,5,1024;key_padding_mask:5,90" \
            --soc_version=Ascend${chip_name} \
            --precision_mode=allow_mix_precision \
            --modify_mixlist=../ops_info.json
         ```

         c.将m2m_decoder.onnx转为m2m_decoder.om。
         
         ```
         bash ../atc_decoder.sh ${chip_name}
         ```
         
         运行成功后在工作目录下生成m2m_encoder.om、m2m_decoder_first_step.om、m2m_decoder.om模型文件。

2. 开始推理验证。

   1. 使用ais-infer提供的Python接口进行推理，相关代码通过'fairseq_m2m100.patch'文件更新在源码中。

   2. 执行推理。

        ```
        export ENCODER_OM=./m2m_encoder.om
        export DECODER_FIRST_OM=./m2m_decoder_first_step.om
        export DECODER_OM=./m2m_decoder.om

        fairseq-generate data_bin \
           --batch-size 1 \
           --path 418M_last_checkpoint.pt \
           --fixed-dictionary model_dict.128k.txt \
           -s en \
           -t zh \
           --remove-bpe 'sentencepiece' \
           --beam 5 \
           --task translation_multi_simple_epoch \
           --lang-pairs language_pairs_small_models.txt \
           --decoder-langtok \
           --encoder-langtok src \
           --gen-subset test
        ```

        参数说明：环境变量`ENCODER_OM`、`DECODER_FIRST_OM`、`DECODER_OM`分别指定对应的om模型路径。fairseq-generate参数说明通过`fairseq-generate -h`命令查看。

   3. 精度验证。

      在上一步执行推理过程中，直接将om的推理结果进行后处理，对数据的推理结果显示如下。其中S-14表示原始英文语句；T-14表示原始英文语句的标签；H-14表示om模型翻译结果。将om模型的翻译结果与标签比对，验证模型的推理效果。

      ```
      S-14    __en__ New Delhi:
      T-14    新德里:
      H-14    -1.6589527130126953     新德里:
      D-14    -1.6589527130126953     新德里:
      P-14    -8.0156 -0.9091 -0.2527 -0.1126 -0.4881 -0.1756
      ```

   4. 性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同om模型的性能，参考命令如下：

        ```
        python3 -m ais_bench --model=m2m_encoder.om --loop=20 --batchsize=1
        python3 -m ais_bench --model=m2m_decoder_first_step.om --loop=20 --batchsize=1
        python3 -m ais_bench --model=m2m_decoder.om --loop=20 --batchsize=1 --dymDims="prev_output_tokens:5,101;65:120,16,100,64;66:120,16,90,64;67:60,90"
        ```

      参数说明：ais_bench参数说明可通过'python3 -m ais_bench -h'命令查看。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

- 注：端到端推理精度请参考`开始推理验证`部分。

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Model | Batch Size   | 数据集 | 精度 | 性能 |
| ---------- | ------                    | --------- | ------ | ---------- | -------|
| Ascend310P | m2m_encoder.om            | 1         | - | - | 224.376 fps |
| Ascend310P | m2m_decoder_first_step.om | 1         | - | - | 82.827 fps |
| Ascend310P | m2m_decoder.om            | 1         | - | - | 141.6 fps |
