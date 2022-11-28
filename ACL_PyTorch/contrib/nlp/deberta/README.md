# Deberta模型-推理指导


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

DeBERTa使用两种新技术改进了BERT和RoBERTa模型。第一种是解耦注意力机制，其中每个单词使用content embedding和position embedding表示，单词之间的注意力权重使用关于其内容和相对位置的解耦矩阵来计算。第二，使用增强的掩码解码器来替换输出softmax层，以预测模型预训练中被Mask的Token。


- 参考实现：

  ```
  url=https://github.com/microsoft/DeBERTa.git
  commit_id=c558ad99373dac695128c9ec45f39869aafd374e
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input_ids    | int32 | batchsize x 256 | ND         |
  | input_mask    | int32 | batchsize x 256 | ND         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 3 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动 

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.11.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/microsoft/DeBERTa.git
   cd DeBERTa
   git reset --hard c558ad99373dac695128c9ec45f39869aafd374e
   patch -p1 < ../deberta.patch
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   执行以下命令，或者从此链接获取[MNLI数据集](https://ascend-pytorch-one-datasets.obs.cn-north-4.myhuaweicloud.com/infer/MNLI/MNLI.zip)

   ```
   cd DeBERTa/experiments/glue
   ./download_data.sh ./ MNLI
   cd ..
   ```

   目录结构如下：

   ```
   |-- MNLI
      |-- original
      |    |-- multinli_1.0_dev_matched.jsonl
      |    |-- multinli_1.0_dev_matched.txt
      |    |-- multinli_1.0_dev_mismatched.jsonl
      |    |-- multinli_1.0_dev_mismatched.txt
      |    |-- multinli_1.0_train.jsonl
      |    |-- multinli_1.0_train.txt
      |-- dev_matched.tsv
      |-- dev_mismatched.tsv
      |-- test_matched.tsv
      |-- dev_matched.tsv
      |-- test_mismatched.tsv
      |-- train.tsv
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行deberta_preprocess.py脚本，完成预处理。

   ```
   python3 deberta_preprocess.py --datasets_path ${dataset_path} --pre_data_save_path ./pre_mnli --batch_size 1 
   ```

   - 参数说明：
     - --datasets_path：数据集路径  
     - --pre_data_save_path：预处理后的 bin 文件存放路径  
     - --batch_size 数据batch size
   
   > **说明：**  
   > 在预处理代码里,tokenizers = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base"),需要从网上下载相关文件，可能会存在无法下载的问题。
   > 解决方法：下载[pre_deberta.zip](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/nlp/Deberta/pre_deberta.zip),将里面的文件放在根目录~/.cache/huggingface/transformers下,预处理可正常运行


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       [pytorch.model-073631.bin](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/nlp/Deberta/pytorch.model-073631.bin)

   2. 导出onnx文件。

      1. 使用pth2onnx.py导出onnx文件。

         ```
         python3.7 pth2onnx.py --init_model ./pytorch.model-073631.bin --onnx_path ./deberta.onnx --config ./model_config.json
         ```
         其中"init_model"表示模型加载权重的地址和名称,"onnx_path"表示转换后生成的onnx模型的存储地址和名称,[model_config.json](./model_config.json)是模型训练的时候在预训练config的基础上加上mnli任务的config拼接得到的中间文件，用于初始化模型。

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
         atc --framework=5 \
         --model=./deberta.onnx \
         --output=./deberta_bs${batch_size} \
         --input_format=ND \
         --input_shape="input_ids:${batch_size},256;input_mask:${batch_size},256" \
         --soc_version=Ascend${chip_name} \
         --log=error
         ```

         说明：\$\{batch\_size\} 表示生成不同 batch size 的 om 模型

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

         运行成功后生成<u>***deberta_bs${batch_size}.om***</u>模型文件。

2. 开始推理验证。

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
        mkdir result
        mkdir result/${dataset_version}
        python3 ais_infer.py --model ./deberta_bs${batch_size}.om --input ./pre_mnli/${dataset_version}/input_ids/,./pre_mnli/${dataset_version}/input_mask/ --output ./result/${dataset_version}  
        ```

        -   参数说明：

             -   --model：om 模型路径。
             -   --input：预处理后的 bin 文件存放路径。
             -   --output：输出文件存放路径。
      
        \${dataset_version} 表示使用哪种数据集类型，取值 match 或者 mismatch  
        \${batch_size} 表示推理使用模型的 batch size

        推理后的输出默认在当前目录result下。

        >**说明：** 
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   3. 精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

      ```
      python3.7 deberta_postprocess.py --datasets_path ${dataset_path}/ --bin_file_path ./result/${dataset_version}/*/ --dataset_version ${dataset_version} --eval_save_path ./result --eval_save_file eval_bs${batch_size}_${dataset_version}.txt
      ```

      - 参数说明：

        - --bin_file_path：生成推理结果所在路径
        - --dataset_version：表示使用哪种数据集类型，取值 match 或者 mismatch
        - --output：输出文件存放路径 
        - --eval_save_path：输出精度数据文件所在的路径
        - --eval_save_file：输出精度数据文件名

   4. 性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7 ${ais_infer_path}/ais_infer.py --model=${om_model_path} --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om 模型路径
        - --batchsize：batch大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 模型    | 官网精度 | 310P 精度 | 基准性能 | 310P 性能 |
| ------- | ------- | -------- | -------- | -------- |
| Deberta bs1  | M=90.5/MM=90.5 | M=90.46/MM=90.71 | 2.849fps | 4.055fps |
| Deberta bs4  | M=90.5/MM=90.5 | M=90.46/MM=90.71 | 2.859fps | 4.692fps |
| Deberta bs8  | M=90.5/MM=90.5 | M=90.46/MM=90.71 | 2.821fps  | 4.639fps |
| Deberta bs16  | M=90.5/MM=90.5 | M=90.46/MM=90.71 | 2.807fps | 4.681fps |
| Deberta bs32  | M=90.5/MM=90.5 | M=90.46/MM=90.71 | 2.935fps | 4.619fps |
| Deberta bs64  | M=90.5/MM=90.5 | M=90.46/MM=90.71 | 3.065fps | 5.027fps  |