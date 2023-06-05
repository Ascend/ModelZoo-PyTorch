# GNMT模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


******

  
# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

GNMT是一个端到端机器翻译系统，它解决了NMT训练速度慢、很难处理生词、无法完全覆盖长句等问题。GNMT包含了encoder-decoder结构，encoder-decoder中都包含8层LSTM网络，encoder和decoder内部均使用残差连接，并且使用了attention。谷歌对NMT做了一些改进：1）为了提高翻译效率，在翻译过程中使用低精度的算法。2）为了解决输入的生词，在输入和输出中使用了sub-word units。3）在beam search中加入关于长度的正则化项，和一个用于鼓励生成的句子尽可能覆盖所有source sentence的惩罚项。


- 参考实现：
    ```
    url=https://github.com/NVIDIA/DeepLearningExamples.git
    commit_id=90f94bd77e8d4c75ad9cc25f03fcf9f09af28a63
    model_name=GNMT
    branch=master
    code_path=PyTorch/Translation/GNMT
    ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

    | 输入数据      | 数据类型 | 大小   | 数据排布格式 |
    | ------------- | -------- | ------ | ------------ |
    | input_encoder | INT32    | 1 x 30 | ND           |
    | input_enc_len | INT32    | 1      | ND           |
    | input_decoder | INT32    | 1 x 1  | ND           |
  
    注：此模型当前仅支持 batchsize=1。


- 输出数据

    | 输出数据    | 大小   | 数据类型 | 数据排布格式 |
    | ----------- | ------ | -------- | ------------ |
    | translation | 1 x 30 | INT32    | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

    **表 1**  版本配套表

    | 配套                                                         | 版本    | 环境准备指导                                                 |
    | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
    | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN                                                         | 6.0.RC1 | -                                                            |
    | Python                                                       | 3.7.5   | -                                                            |
    | PyTorch                                                      | 1.12.0  | -                                                            |
    | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

1. 获取开源代码。
   ```bash
   git clone https://github.com/NVIDIA/DeepLearningExamples.git -b master
   cd DeepLearningExamples
   git reset --hard 90f94bd77e8d4c75ad9cc25f03fcf9f09af28a63
   cd ..
   ```

2. 安装依赖。
   ```bash
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

    运行gnmt_data.sh脚本下载newstest2014数据集。
    ```bash
    bash gnmt_data.sh
    ```

2. 获取权重文件。  
    将权重文件[gnmt.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/GNMT/PTH/gnmt.pth)下载到本地后上传到当前工作目录。

3. 数据预处理。

    数据预处理将原始数据集转换为模型输入的数据。

    执行gnmt_preprocess.py脚本，完成预处理。
    ```bash
    python3 gnmt_preprocess.py \
        --model_path ./gnmt.pth \
        --data_path ./data \
        --pre_data_save_path ./pre_data \
        --max_seq_len 30
    ```

    -  参数说明：
        -  --model_path：模型权重文件路径。
        -  --data_path：测试集路径。
        -  --pre_data_save_path：预处理后bin文件保存路径。
        -  --max_seq_len：最大文本长度，默认30。


## 模型推理<a name="section741711594517"></a>

1. 模型修改。

    应用补丁，修改模型代码。
    ```bash
    cd DeepLearningExamples
    patch -p1 < ../gnmt.patch
    cd ..
    ```

1. 模型转换。

    使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

    1. 导出onnx文件。

        1. 使用pth2onnx.py导出onnx文件。

            运行pth2onnx.py脚本。
            ```bash
            python3 pth2onnx.py \
                --model ./gnmt.pth \
                --onnx_dir ./ \
                --max_seq_len 30
            ```

            -  参数说明：
                -  --model：模型权重文件路径。
                -  --onnx_dir：onnx文件的保存路径。
                -  --max_seq_len：最大文本长度，默认30。
   
            获得`gnmt_msl30.onnx`文件。
   
        2. 优化ONNX文件。
            ```bash
            python3 -m onnxsim gnmt_msl30.onnx gnmt_msl30_sim.onnx
            ```

            获得`gnmt_msl30_sim.onnx`文件。

    2. 使用ATC工具将ONNX模型转OM模型。
   
        1. 配置环境变量。
            ```bash
            source /usr/local/Ascend/ascend-toolkit/set_env.sh
            ```
   
            > **说明：** 
            >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见[《CANN 开发辅助工具指南 \(推理\)》](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)。
   
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
            ```bash
            atc --framework=5 \
                --model=gnmt_msl30_sim.onnx \
                --output=gnmt_msl30_sim \
                --input_format=ND \
                --input_shape="input_encoder:1,30;input_enc_len:1;input_decoder:1,1" \
                --soc_version=Ascend${chip_name} \
                --fusion_switch_file=fusion_switch.cfg \
                --log=error
            ```
         
            -  参数说明：
                -  --model：为ONNX模型文件。
                -  --framework：5代表ONNX模型。
                -  --output：输出的OM模型。
                -  --input_format：输入数据的格式。
                -  --input_shape：输入数据的shape。
                -  --log：日志级别。
                -  --soc_version：处理器型号。
                -  --fusion_switch_file：融合规则开关配置文件路径。
         
            运行成功后生成 gnmt_msl30_sim.om 模型文件。

2. 开始推理验证。

    1. 安装ais_bench推理工具。  
        请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

    2. 执行推理。
        ```bash
        python3 -m ais_bench \
            --model=./gnmt_msl30_sim.om \
            --input=./pre_data/input_encoder/,./pre_data/input_enc_len,./pre_data/input_decoder \
            --output=. \
            --output_dirname=result \
            --outfmt=BIN
        ```

        -  参数说明：
            -  --model：om文件路径。
            -  --input：输入名及文件路径。
            -  --output：输出路径。
            -  --outfmt：输出文件格式。
            -  --output_dirname: 输出结果保存文件夹。

        推理后的输出默认在当前目录result下。


    3. 精度验证。

        调用脚本进行后处理，可以获得翻译结果，并得到BLEU分数，译文保存在res_data/pred_sentences.txt中。
        ```bash
        python3 gnmt_postprocess.py \
            --model_path ./gnmt.pth \
            --bin_file_path ./result \
            --res_file_path ./res_data \
            --pre_file_path ./pre_data
        ```

        -  参数说明：
            -  --bin_file_path：ais_bench自动生成的目录名。
            -  --res_file_path：推理结果保存在该目录的 pred_sentences.txt 文件中。
            -  --pre_file_path：预处理文件目录。

    4. 性能验证。

        使用ais_bench推理工具进行纯推理，获得性能数据。
        ```bash
        python3 -m ais_bench \
            --model=./gnmt_msl30_sim.om \
            --loop 20 \
            --batchsize 1
        ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| Ascend 310P3 | 1 | newstest2014 | BLEU：22.67 | 24.55 fps |
