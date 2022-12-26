# bert-large-NER模型-推理指导

-   [概述](#ZH-CN_TOPIC_0000001172161501)
  
-   [推理环境准备](#ZH-CN_TOPIC_0000001126281702)
  
-   [快速上手](#ZH-CN_TOPIC_0000001126281700)
  
    -   [获取源码](#section4622531142816)
      
    -   [准备数据集](#section183221994411)
      
    -   [模型推理](#section741711594517)
    
-   [模型推理性能](#ZH-CN_TOPIC_0000001172201573)
  
-   [配套环境](#ZH-CN_TOPIC_0000001126121892)
  
    ----------
    

# 概述

**bert-large-NER**  是一个经过微调的 BERT 模型，可用于**命名实体识别**，并为 NER 任务实现**一流的性能**。它已经过训练，可以识别四种类型的实体：位置（LOC），组织（ORG），人员（PER）和杂项（杂项）。

具体而言，此模型是一个*bert-large-cased*模型，在标准  [CoNLL-2003 命名实体识别](https://www.aclweb.org/anthology/W03-0419.pdf)数据集的英文版上进行了微调。如果要在同一数据集上使用较小的 BERT 模型进行微调，也可以使用[**基于 NER 的 BERT**](https://huggingface.co/dslim/bert-base-NER/)  版本。

* 参考实现：

  ```
  url = https://huggingface.co/dslim/bert-large-NER
  commit_id =  95c62bc0d4109bd97d0578e5ff482e6b84c2b8b9
  model_name = bert-large-NER
  ```

## 输入输出数据

- 输入数据

  | 输入数据       | 数据类型 | 大小            | 数据排布格式 |
  | -------------- | -------- | --------------- | ------------ |
  | input_ids      | int64    | batchsize x 512 | ND           |
  | attention_mask | int64    | batchsize x 512 | ND           |
  | token_type_ids | int64    | batchsize x 512 | ND           |


- 输出数据

  | 输出数据 | 大小                 | 数据类型 | 数据排布格式 |
  | -------- | -------------------- | -------- | ------------ |
  | logits   | batchsize x 512  x 9 | FLOAT32  | ND           |


# 推理环境准备

-   该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套       | 版本    | 环境准备指导                                                 |
| ---------- | ------- | ------------------------------------------------------------ |
| 固件与驱动 | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN       | 6.0.RC1 | -                                                            |
| Python     | 3.7.5   | -                                                            |
| Pytorch    | 1.12.0  | -                                                            |

# 快速上手

## 获取源码

1.  获取开源模型。
  
     1. 安装git-lfs（后续使用transformer[onnx]从开源权重导出onnx模型文件时，必须保证安装git-lfs）
     
        1. git-lfs需要git版本≥1.8.2
     
           ```
           git --version
           ```
     
        2. 在linux上下载git-lfs
     
           ```
           curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
           sudo apt-get install git-lfs
           ```
     
        3. 安装git-lfs
     
           ```
           git-lfs install
           ```
     
     2. 获取权重文件
     
        ```
        git clone https://huggingface.co/dslim/bert-large-NER
        ```
     
2.  安装依赖。
  
  ```
  pip install -r requirements.txt
  ```
  
  

## 准备数据集

1.  获取原始数据集。

   数据集名称： **CoNll-2003**：信息抽取-命名实体识别（NER）
   下载链接：[CoNLL 2003 (English) Dataset | DeepAI](https://data.deepai.org/conll2003.zip)
   目录结构：./bert-large/conll2003
   
   ```
   conll2003
   ├── metadata
   ├── train.txt
   ├── test.txt
   └── valid.txt
   ```

2.  数据预处理，将原始数据集转换为模型输入的数据。
  
    生成om模型需要的推理数据，完成预处理：
    ```
    python bin_create.py --input_file='./conll2003/test.txt' --model_name_from_hub='./bert-large-NER'
    ```
    * --input_file：输入数据集
    * --model_name_from_hub：所使用的hugging face 开源模型名，需要调用其中的的字典进行数据生成 
    * 运行成功后生成：input_ids、attention_mask、token_type_ids三个文件夹，作为om模型的输入。.anno文件记录token对应的label。

## 模型推理

 1. 模型转换。
    使用 transformer[onnx] 将模型权重文件转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
    
    1. 导出onnx文件。
    
        1.  使用transformers[onnx]导出onnx文件
          
            ``` 
            pip install transforemers[onnx]
            ```
            
            ```
            python -m transformers.onnx --model=bert-large-NER --feature=token-classification onnx/
            ```
            
            * --model：hugging face上下载的开源模型
            * --feature：用于导出模型的特征类型
            * onnx/：保存导出的onnx模型的路径
            
              运行结束后生成model.onnx保存在./onnx文件夹下。
            
        2.  对onnx模型进行优化。
    
            1. 安装改图依赖
    
               ```\
               git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
               cd MagicONNX && git checkout 45f4fd50a7e7ca76ee90365fec1a907b25ffa8e8
               pip3 install . && cd ..
               ```
    
            2. 简化并修改onnx文件
    
               ```
               # 简化onnx文件 其中seq_len=512，bs为批次大小
               python -m onnxsim model.onnx bert_large_bs${bs}_sim.onnx --input-shape "input_ids:${bs},${seq_len}" "attention_mask:${bs},${seq_len}" "token_type_ids:${bs},${seq_len}"
               # 修改onnx文件
               python fix_onnx.py bert_large_bs${bs}_sim.onnx bert_large_bs${bs}_fix.onnx
               ```
    
               * bs为批次大小；seq_len=512为序列长度
               * 运行结束后生成简化后固定输入shape的onnx模型，并对其进行修改，生成修改后的onnx模型：***bert_large_bs${bs}_fix.onnx***模型文件，随后将修改后的onnx模型转om。
    
               
    
    2. 使用ATC工具将ONNX模型转OM模型。
    
	    1.  配置环境变量。
          
             source /usr/local/Ascend/ascend-toolkit/set_env.sh
             
            
            > **说明：**  该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 (推理)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
            
        2. 执行命令查看芯片名称（${chip_name}）。
    
             ```
             npu-smi info   #该设备芯片名为Ascend310P3 
             该设备回显为：
             +--------------------------------------------------------------------------------------------+
             | npu-smi 22.0.0                       Version: 22.0.3                                       |
             +-------------------+-----------------+------------------------------------------------------+
             | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
             | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
             +===================+=================+======================================================+
             | 0       310P3     | OK              | 17.1         56                0    / 0              |
             | 0       0         | 0000:86:00.0    | 0            3292 / 21527                            |
             +===================+=================+======================================================+
             
             ```
             
        3. 执行ATC命令。
        
           执行推理前，手动创建推理结果保存目录：
           
           ```
           mkdir -p ./bert-large-OUT/bs${bs}
           ```
           
           使用ATC工具进行推理：
           
           ```
           # 其中seq_len=512，bs为批次大小
           atc --framework=5 --model=bert_large_bs${bs}_fix.onnx --output=bert_large_bs${bs}_fix --input_shape="input_ids:${bs},${seq_len};attention_mask:${bs},${seq_len};token_type_ids:${bs},${seq_len}" --soc_version=${chip_name} --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
           ```
           
           -   参数说明：
                -   --model：为ONNX模型文件。
                -   --framework：5代表ONNX模型。
                -   --output：输出的OM模型。
                -   --input_shape：输入数据的shape。输入数据有三条，均为batch_size*${seq_len}，其中sequence序列长度为512。
                -   op_select_implmode：**high_performance**，表示网络模型中算子采用高性能实现模式。
                -   optypelist_for_implmode：设置optype列表中算子的实现方式，该参数需要与[--op_select_implmode](https://support.huawei.com/enterprise/zh/doc/EDOC1100232270?section=j014#ZH-CN_TOPIC_0000001161902398)参数配合使用。
                -   --soc_version：处理器型号。
                    运行成功后生成***bert_large_bs${bs}_fix.om***模型文件。
    
 2.  使用om模型进行推理验证。
   
     a. 准备推理工具ais_bench。
      
       请点击本链接进行安装ais_bench推理工具，以及查看具体使用方法(https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)  
     
     b. 执行推理
      
       1. 性能测试
        
          ```
          # 性能测试
          python -m ais_bench --model ./bert_large_bs${bs}_fix.om --batchsize ${bs} --loop 50
          ```
        
          参数说明：
        
          * --model：om文件路径
          
          * --batchsize：批处理大小
          
            执行结束输出使用bert_large_bs${bs}_fix.om模型进行推理时所需的throughoutput。
        
       2. 精度测试
        
          ```
          # 精度测试
          python -m ais_bench --model ./bert_large_bs${bs}_fix.om --input "./bert_bin/bert_bin_2022xxxx-xxxxxx/input_ids,./bert_bin/bert_bin_2022xxxx-xxxxxx/attention_mask,./bert_bin/bert_bin_2022xxxx-xxxxxx/token_type_ids" --output ./bert-large-OUT/bs${bs} --batchsize ${bs} --outfmt NPY
          ```
        
          参数说明：
        
          * --model：om文件路径
          
          * --batchsize：指定batchsize大小
          
          * --input：模型的输入，input_ids、attention_mask、token_type_ids三个文件夹
          
          * --output：输出指定在./bert-large-OUT/bs${bs}下
          
          * --outfmt：推理结果保存格式
          
            执行结束输出保存在 ./bert-large-OUT/bs${bs}下。
      
    c. 精度验证
    
       ```
       # 调用脚本将推理生成数据与数据集标签比对，可以获得Accuracy数据，结果保存在result.json中
       python bert_metric.py --result_dir ./bert-large-OUT/bs${bs}/2022_xx_xx-xx_xx_xx --anno_file ./bert_bin/bert_bin_2022xxxx-xxxxxx.anno
       ```
    
       参数说明：
    
       * --result_dir：om模型的推理结果保存的路径
       
       * --anno_file：保存推理数据的token对应label           
       
         ​                                                                                                

# 模型推理性能&精度

1. 性能对比

| Batch Size | 310P性能 | T4性能 | 310P/T4 |
| :--------: | :----: | :-----: | :-----: |
|   1   | 63.3062 |  57.6312  |  1.098 |
|   4   | 70.6858 |  61.5656  |  1.148 |
|   8   | 74.5971 |  60.7644  |  1.227 |
|   16   | 73.7290 |  64.0751  |  1.150 |
|   32   | 73.6084 |  64.7773  |  1.136 |
|   64   | 71.1308 |  63.9418  |  1.112 |

* 注：T4上的性能计算为1000/(GPU Compute mean/batch)

2. 精度对比

|      模型      | Batch Size | acc(NPU) | acc(开源仓) |
| :------------: | :--------: | :-----------: | :--------------: |
| bert_large_NER |     1      |     90.74     |      91.2        |
| bert_large_NER |     4      |     90.74     |      91.2        |
| bert_large_NER |     8      |     90.74     |      91.2        |
| bert_large_NER |     16     |     90.74     |      91.2        |
| bert_large_NER |     32     |     90.74     |      91.2        |
| bert_large_NER |     64     |     90.74     |      91.2        |





