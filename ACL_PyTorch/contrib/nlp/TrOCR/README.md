# TrOCR模型-推理指导


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

TrOCR是一种端到端的文本识别方法，具有预先训练好的图像Transformer和文本Transformer模型，它利用Transformer架构进行图像理解和词条级文本生成。


- 参考实现：
    ```
    url=https://github.com/microsoft/unilm.git
    commit_id=97d18544e207159c53cec40bd9767746df5443a0
    model_name=TrOCR
    branch=master
    code_path=/trocr
    ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

    | 输入数据      | 数据类型 | 大小   | 数据排布格式 |
    | ------------- | -------- | ------ | ------------ |
    | input | INT32    | 1 x 3 x 384 x 384 | NCHW           |
  
    注：此模型当前仅支持 batchsize=1。


- 输出数据

    | 输出数据    | 大小   | 数据类型 | 数据排布格式 |
    | ----------- | ------ | -------- | ------------ |
    | cand_bbsz_idx_out | 31 x 1 x 20 | INT32    | ND           |
    | eos_mask_out | 31 x 1 x 20 | BOOL    | ND           |
    | cand_scores_out | 31 x 1 x 20 | FLOAT32    | ND           |
    | tokens_out | 31 x 10 x 202 | INT32    | ND           |
    | scores_out | 31 x 10 x 201 | FLOAT32    | ND           |
    | attn_out | 31 x 10 x 578x202 | FLOAT32    | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

    **表 1**  版本配套表

    | 配套                                                         | 版本    | 环境准备指导                                                 |
    | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
    | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN                                                         | 6.0.RC1 | -                                                            |
    | Python                                                       | 3.7.5   | -                                                            |
    | PyTorch                                                      | 1.12.0  | -                                                            |
    | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

1. 获取开源代码。
    ```bash
    git clone https://github.com/microsoft/unilm.git
    cd unilm/trocr
    git reset --hard 97d18544e207159c53cec40bd9767746df5443a0
    cd ../..
    git clone https://github.com/pytorch/fairseq
    cd fairseq
    git reset --hard 806855bf660ea748ed7ffb42fe8dcc881ca3aca0
    cd ..
    ```

2. 安装依赖。
    ```bash
    pip3 install pybind11
    pip3 install -r requirements.txt
    patch -p1 < trocr.patch
    cd fairseq
    pip3 install --editable ./
    cd ..
    ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

    获取[IAM数据集](https://github.com/microsoft/unilm/tree/master/trocr)并解压数据集至当前目录。
    ```bash
    wget https://layoutlm.blob.core.windows.net/trocr/dataset/IAM.tar.gz
    tar zxvf IAM.tar.gz
    ```

    注：若链接下载失败，请在链接的后面添加以下后缀重试：
    ```
    ?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D
    ```

2. 获取权重文件。  
    
    ```bash
    wget https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-small-handwritten.pt
    ```

    注：若链接下载失败，操作方法和上一步骤相同。

3. 数据预处理。

    数据预处理将原始数据集转换为模型输入的数据。

    执行trocr_preprocess.py脚本，完成预处理。
    ```bash
    python3 trocr_preprocess.py --model_path ./trocr-small-handwritten.pt \
                                  --datasets_path ./IAM \
                                  --pre_data_save_path ./pre_data
    ```

    -  参数说明：
        -  --model_path：模型pt文件路径。
        -  --datasets_path：数据集路径。
        -  --pre_data_save_path：预处理后的 bin 文件存放路径。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

    使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

    1. 导出onnx文件。

        1. 使用pth2onnx.py导出onnx文件。

            运行pth2onnx.py脚本。
            ```bash
            python3 pth2onnx.py --model ./trocr-small-handwritten.pt \
                                  --onnx_dir ./
            ```

            -  参数说明：
                -  --model：模型权重文件路径。
                -  --onnx_dir：onnx文件的保存路径。
   
            获得trocr.onnx文件。
   
        2. 优化ONNX文件。
            ```bash
            python3 -m onnxsim trocr.onnx trocr_sim.onnx
            ```

            获得trocr_sim.onnx文件。

    2. 使用ATC工具将ONNX模型转OM模型。
   
        1. 配置环境变量。
            ```bash
            source /usr/local/Ascend/ascend-toolkit/set_env.sh
            ```
   
            > **说明：** 
            >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见[《CANN 开发辅助工具指南 \(推理\)》](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)。
   
        2. 执行命令查看芯片名称（$\{chip\_name\}）。
            ```bash
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
                --model=./trocr_sim.onnx \
                --output=./trocr_bs1 \
                --input_format=NCHW  \
                --input_shape="imgs:1,3,384,384" \
                --soc_version=${chip_name} \
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
            
            运行成功后生成 trocr_s1.om 模型文件。

2. 开始推理验证。

    1. 安装ais_bench推理工具。  
        请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

    2. 推理时，使用 npu-smi info 命令查看 device 是否在运行其它推理任务，提前确保 device 空闲
        ```bash
        # 删除之前冗余的推理文件，创建 out 文件夹
        rm -rf ./out/
        mkdir -p ./out/

        # 推理
        python3 -m ais_bench --model ./trocr_bs1.om \
                --input ./pre_data \
                --output ./out
        ```

        - 参数说明:    
            - --model：om 模型路径      
            - --input：预处理后的 bin 文件存放路径      
            - --output：输出文件存放路径 


    3. 精度验证。

        使用fairseq库的generate功能进行后处理。
        ```bash
        cd unilm/trocr
        $(which fairseq-generate) \
                --data-type STR \
                --user-dir ./ \
                --task text_recognition \
                --input-size 384 \
                --beam 10 \
                --scoring cer2 \
                --gen-subset test \
                --batch-size 1 \
                --path ../../trocr-small-handwritten.pt \
                --results-path ../../result \
                --preprocess DA2 \
                --bpe sentencepiece \
                --sentencepiece-model ./unilm3-cased.model \
                --dict-path-or-url https://layoutlm.blob.core.windows.net/trocr/dictionaries/unilm3.dict.txt \
                --fp16 \
                ../../IAM
        cd ../..
        ```
        - 参数说明:    
            - --data-type：数据集类型
            - --user-dir：包含Python模块的路径      
            - --task：任务类型    
            - --beam：beam大小
            - --scoring：评估的指标
            - --gen-subset：要建立的数据集子集
            - --batch-size：Batch大小
            - --path：模型权重路径
            - --result-path：验证结果存放路径
            - --preprocess：测试数据的预处理方法
            - --bpe：tokenize方式
            - --sentencepiece-model：sentencepiece编码模型
            - --dict-path-or-url：字典的路径或者URL
            - --fp16：使用FP16精度
            - data：数据集路径


        后处理精度验证结果保存在当前目录下的result文件夹中。

        打印精度结果
        ```bash
        cat result/generate-test.txt | tail -1
        ```

    4. 性能验证。

        使用ais_bench推理工具进行纯推理。使用同一输入进行性能测试，与基准性能对比：
        ```bash
        python3 -m ais_bench --model ./trocr_bs1.om --loop 100
        ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| :-------: | :--------------: | :--------: | :--------: | :-------------: |
| Ascend 310P3 | 1 | IAM | 4.25(Cased CER) | 8.79 fps |
