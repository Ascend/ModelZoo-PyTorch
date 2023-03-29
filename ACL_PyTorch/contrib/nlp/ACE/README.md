# ACE模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

******

  
# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

ACE是一个用于自动搜索结构化预测任务的良好嵌入连接的框架。

- 参考实现：
    ```
    url=https://github.com/Alibaba-NLP/ACE.git
    commit_id=bafa07c39b8a5b2753a770362dc42f2a0526c4d3
    model_name=ACE
    branch=master
    code_path=PyTorch/nlp/ACE
    ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

    | 输入数据            | 数据类型              | 大小                | 数据排布格式                  |
    | ----------------- | ----------------- |-------------------------| ------------ |
    |  sentence_tensor  | FLOAT32           | batchsize x 124 x 24876 | ND                      |
    |  lengths_tensor   | INT32             | batchsize         | ND                      |



- 输出数据

    | 输出数据 | 大小                   | 数据类型    | 数据排布格式 |
    |----------------------|---------| -------- | ------------ |
    | features    | batchsize x 124 x 20 | FLOAT32 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

    **表 1**  版本配套表

    | 配套                                                         | 版本      | 环境准备指导                                                 |
    |---------| ------- | ------------------------------------------------------------ |
    | 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN                                                         | 6.0.RC1 | -                                                            |
    | Python                                                       | 3.7.5   | -                                                            |
    | PyTorch                                                      | 1.8.1   | -                                                            |
    | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取开源代码。
   ```
   git clone https://github.com/Alibaba-NLP/ACE.git -b main
   cd ACE
   git reset --hard bafa07c39b8a5b2753a770362dc42f2a0526c4d3
   patch -p1 < ../ACE.patch
   cd ..
   ```

2. 安装依赖。
   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。
    将[数据集](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com:443/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/nlp/ACE/datasets.tar?AccessKeyId=4WKXHKTRCNZGLVNUBZWO&Expires=1699870153&Signature=zeXA1xuFtlB3Awk4YSXdu9cv1E8%3D)
    下载后，在用户的根目录创建`.flair/`目录，将数据集解压后的`datasets`目录移到`.flair/`下。
    解压后数据集目录结构如下:
    ```
      datasets
          |-- conll_03_english
          |    |-- train.txt
          |    |-- testa.txt
          |    |-- testb.txt
    ```
    以root用户为例，在`/root`下创建`.flair/`目录，将数据集解压后的`datasets`目录移到`/root/.flair/`中。

2. 获取权重文件。  
    将权重文件[ace.pth](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com:443/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/nlp/ACE/model.tar?AccessKeyId=4WKXHKTRCNZGLVNUBZWO&Expires=1699870181&Signature=ZeCBQbyN0x4N5hihdWC1TJtK%2BhI%3D)
    下载到本地后解压到ACE/resources/taggers/目录。
3. 数据预处理。

    数据预处理将原始数据集转换为模型输入的数据。

    执行ace_preprocess.py脚本，完成预处理。
    ```
    python3.7 ace_preprocess.py \
        --config ./ACE/config/doc_ner_best.yaml \
        --pre_data_save_path ./pre_data_bs${batch_size} \
        --batch_size ${batch_size}
    ```

    -  参数说明：
        - --config：模型配置文件路径。
        - --pre_data_save_path：预处理后bin文件保存路径。
        - --batch_size：生成数据集对应的batch size。

> **说明：**  
> 在预处理代码里`student = config.create_student(nocrf=False)`,需要从网上下载相关文件，可能会存在无法下载的问题。
> 解决方法：  
> 1.下载[transformers.tar](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com:443/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/nlp/ACE/transformers.tar?AccessKeyId=4WKXHKTRCNZGLVNUBZWO&Expires=1699870200&Signature=V8hZXtz1S%2B5V3FzeibabtTzT2jg%3D)
> 后，解压到~/.cache/torch/,  
> 2.下载[embeddings.tar](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com:443/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/nlp/ACE/embeddings.tar?AccessKeyId=4WKXHKTRCNZGLVNUBZWO&Expires=1699870167&Signature=RElXgboeJNtrjFDs4aaaQ5X/Bn4%3D)
> 后，解压到~/.flair/,  
> 3.下载[allennlp.tar](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com:443/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/nlp/ACE/allennlp.tar?AccessKeyId=4WKXHKTRCNZGLVNUBZWO&Expires=1699870104&Signature=I5S2NsebF7bFD0c1yvUs7KhEy7U%3D)
> 后，解压到~/，  
> 4.下载[bert.tar](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com:443/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/nlp/ACE/bert.tar?AccessKeyId=4WKXHKTRCNZGLVNUBZWO&Expires=1699870135&Signature=IJO%2BLzco/dkiKAj9eP3Qylb1PRs%3D)
> 后，将里面的文件解压后放在当前目录（ModelZoo-PyTorch/ACL_PyTorch/contrib/nlp/ACE）目录下。  
> 预处理可正常运行

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

    使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

    1. 导出onnx文件。

        1. 使用pth2onnx.py导出onnx文件。

            运行pth2onnx.py脚本。
            ```
            python3.7 pth2onnx.py \
            --config ./ACE/config/doc_ner_best.yaml \
            --batch_size ${batch_size} \
            --onnx_dir ./
            ```

            -  参数说明：
                -  --config：模型权重文件路径。
                -  --batch_size：生成数据集对应的batch size。
                -  --onnx_dir：onnx文件的保存路径。
   
            获得ace_bs${batch_size}.onnx文件。
   
        2. 优化ONNX文件。
            ```
            python3.7 -m onnxsim ace_bs${batch_size}.onnx ace_bs${batch_size}_sim.onnx
            ```

            获得ace_bs${batch_size}_sim.onnx文件。

        3. 使用ATC工具将ONNX模型转OM模型。
   
            1. 配置环境变量。
                ```
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
                ```
                atc --framework=5 \
                    --model=./ace_bs${batch_size}_sim.onnx \
                    --output=./ace_bs${batch_size}_sim \
                    --input_format=ND \
                    --input_shape="sentence_tensor:${batch_size},124,24876;lengths_tensor:${batch_size}" \
                    --soc_version=Ascend${chip_name} \
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
         
                运行成功后生成 ace_bs${batch_size}_sim.om 模型文件。

2. 开始推理验证。

    1. 使用ais-infer工具进行推理。  
        ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

    2. 执行推理。
        ```
        mkdir out_data_bs${batch_size}
        python3.7 ${ais_infer_path}/ais_infer.py \
            --model=./ace_bs${batch_size}_sim.om \
            --input=./pre_data_bs${batch_size}/sentence/,./pre_data_bs${batch_size}/lengths \
            --output=./out_data_bs${batch_size}/ \
            --outfmt=BIN
            --batchsize ${batch_size}
        ```

        -  参数说明：
            - --model：om文件路径。
            - --input：输入名及文件路径。
            - --output：输出路径。
            - --outfmt：输出文件格式。
            - --batch_size 生成数据集对应的batch size。

        推理后的输出默认在当前目录out_data_bs${batch_size}下。

        >**说明：** 
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见[《ais_infer 推理工具使用文档》](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)。

    3. 精度验证。

        调用脚本进行后处理，可以获得翻译结果，并得到分数，译文保存在res_data/accuracy.txt中。
        ```
        python3.7 ace_postprocess.py \
            --config ./ACE/config/doc_ner_best.yaml \
            --bin_file_path ./out_data_bs${batch_size}/2022_xx_xx-xx_xx_xx/ \
            --batch_size ${batch_size}
            --res_file_path ./res_data
        ```

        - 参数说明：
            - --config：模型配置文件路径。
            - --bin_file_path：ais_infer自动生成的目录名。
            - --batch_size：生成数据集对应的batch size。
            - --res_file_path：推理结果保存在该目录的 accuracy.txt 文件中。

    4. 性能验证。

        使用ais-infer工具进行纯推理，获得性能数据。
        ```
        python3.7 ${ais_infer_path}/ais_infer.py \
            --model=./ace_bs${batch_size}_sim.om \
            --loop 50
            --batchsize ${batch_size}
        ```
        - 参数说明：
            - --model：om模型文件路径。
            - --loop：循环次数。
            - --batch_size：生成数据集对应的batch size。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集                                 | 精度    | 性能         |
| --------- |------------|-------------------------------------|-------|------------|
| Ascend 310P3 | 1          | CoNLL 2003 English (document-level)   | 92.57 | 0.712 fps  |
| Ascend 310P3 | 4          | CoNLL 2003 English (document-level)   | 92.56 | 2.884 fps  |
| Ascend 310P3 | 8          | CoNLL 2003 English (document-level)   | 92.52 | 5.720 fps  |
| Ascend 310P3 | 16         | CoNLL 2003 English (document-level)   | 92.53 | 11.477 fps |
| Ascend 310P3 | 32         | CoNLL 2003 English (document-level)   | 92.54 | 11.407 fps |
| Ascend 310P3 | 64         | CoNLL 2003 English (document-level)   | 92.54 | 11.330 fps |
