# VilBert模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

  ViLBERT(Vision-and-Language BERT)模型可以学习视觉内容和文本内容的与特定任务无关的联合表征。ViLBERT在将BERT由单一的文本模态扩展为多模态双流模型。文本流和视觉流通过注意力Transformer层进行交互。

- 参考实现：

  ```
  url=https://github.com/allenai/allennlp.git
  commit_id=80fb6061e568cb9d6ab5d45b661e86eb61b92c82
  url=https://github.com/allenai/allennlp-models.git
  commit_id=b1f372248c17ad12684d344955fbcd98e957e77e
  model_name=vilbert
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据        | 数据类型 | 大小                      | 数据排布格式 |
  | --------        | -------- | ------------------------- | ------------ |
  | box_features    | FLOAT32  | batch_size x box_num x 1024        | ND           |
  | box_coordinates | FLOAT32  | batch_size x box_num x 4           | ND           |
  | box_mask        | FLOAT32  | batch_size x box_num               | ND           |
  | token_ids       | INT64    | batch_size x seq_len               | ND           |
  | mask            | BOOL     | batch_size x seq_len               | ND           |
  | type_ids        | INT64    | batch_size x seq_len               | ND           |

- 输出数据

  | 输出数据 | 数据类型 | 大小      | 数据排布格式 |
  | -------- | -------- | --------  | ------------ |
  | logits   | FLOAT32  | batch_size x 10026 | ND           |
  | probs    | FLOAT32  | batch_size x 10026 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动:

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.12.0 | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master         # 切换到对应分支
   cd ModelZoo-PyTorch/ACL_PyTorch/built-in/nlp/VilBert_for_Pytorch/              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   安装模型依赖:

   ```
   git clone https://github.com/allenai/allennlp.git
   cd allennlp && git checkout 80fb6061e568cb9d6ab5d45b661e86eb61b92c82
   git apply ../allennlp.patch
   pip3 install -r requirements.txt
   python3 setup.py install && cd ..
   git clone https://github.com/allenai/allennlp-models.git
   cd allennlp-models && git checkout b1f372248c17ad12684d344955fbcd98e957e77e
   git apply ../allennlp-models.patch
   pip3 install -r requirements.txt
   python3 setup.py install && cd ..
   git clone https://gitee.com/Ronnie_zheng/MagicONNX
   cd MagicONNX && git checkout dev
   pip3 install . && cd ..
   ```

   回到原仓目录，安装默认依赖：

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>


   使用Balanced Real Images数据集，下载链接：https://visualqa.org/download.html

   仅下载**Validation images**（其它部分会自动下载），将下载的数据集解压到val2014目录，目录结构如下：


   ```
   val2014
   ├── COCO_val2014_000000000136.jpg
   └── ...
   ```

   运行以下命令：

   ```shell
   mkdir -p /net/nfs2.allennlp/data/vision/vqa
   ln -s ${val2014_realpath} /net/nfs2.allennlp/data/vision/vqa/balanced_real
   ```
    
   前处理前依赖模型文件:
   
   [vilbert-vqa-pretrained.2021-03-15.tar.gz权重](https://storage.googleapis.com/allennlp-public-models/vilbert-vqa-pretrained.2021-03-15.tar.gz)
   
   [faster-rcnn预训练模型](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth),拷贝到/root/.cache/torch/hub/checkpoints目录下，如目录不存在则需要创建

   **注意：由于模型仓处理数据时，需要在线下载部分文件，当网络环境不稳定时，可能出现ssl verify错误，可通过以下方法解决：**
   
   ```python
   # 在"/usr/local/python3.7.5/lib/python3.7/urllib/request.py"文件中加入
   import ssl
   ssl._create_default_https_context = ssl._create_unverified_context
   ```

   执行预处理脚本，完成预处理:

   ```
   python3 preprocess.py --archive_file vilbert-vqa-pretrained.2021-03-15.tar.gz --save_dir ./preprocessed_data --pad_len=32 --box_num=100
   ```

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 导出onnx文件。

      1. 使用以下脚本导出onnx文件:

         ```
         python3 export_onnx.py --archive_file vilbert-vqa-pretrained.2021-03-15.tar.gz --save_path vqa-vilbert.onnx
         ```

         获得vqa-vilbert.onnx文件。

      2. 优化ONNX文件。

         ```
         # 以bs1为例
         python -m onnxsim vqa-vilbert.onnx vqa-vilbert_bs1_sim.onnx --input-shape "box_features:1,100,1024" "box_coordinates:1,100,4" "box_mask:1,100" "token_ids:1,32" "mask:1,32" "type_ids:1,32"
         python3 fix_onnx.py vqa-vilbert_bs1_sim.onnx vqa-vilbert_bs1_sim_fix.onnx
         ```

         获得vqa-vilbert_bs1_fix.onnx文件。

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
         # bs1为例
         atc --framework=5 --model=./models/vqa-vilbert_bs1_sim_fix.onnx --output=vqa-vilbert_bs1 --input_format=ND --log=error --soc_version=${chip_name} --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成vqa-vilbert_bs1.om模型文件。

2. 开始推理验证。

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
        mkdir outputs
        python3 -m ais_bench --model vqa-vilbert_bs1.om --input preprocessed_data/box_features,preprocessed_data/box_coordinates,preprocessed_data/box_mask,preprocessed_data/token_ids,preprocessed_data/mask,preprocessed_data/type_ids --output outputs --device 1 --outfmt NPY
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --device：NPU设备编号。


        推理后的输出默认在当前目录outputs下。

        >**说明：** 
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   3. 精度验证。

      调用脚本与GT label，可以获得精度数据:

      ```
      python3 postprocess.py --archive_file=vilbert-vqa-pretrained.2021-03-15.tar.gz --result_dir=outputs/${timestamp}/ --label_dir=preprocessed_data/labels --label_weight_dir=preprocessed_data/label_weights
      ```

      - 参数说明：

        - --archive_file: 模型配置文件

        - --result_dir：为生成推理结果所在路径

        - --label_dir/--label_weight_dir：为标签数据
        
        
   4. 性能验证。
   
      调用ais_ben纯推理测试性能：
      
      ```
      # 以bs1为例
      python3 -m ais_bench --model vqa-vilbert_bs1.om --loop 20 --device 1
      ```
      


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

   精度结果：

   | 模型    | 数据集        | 官网精度                                                  | 310P离线推理精度                                                 |
   |---------|---------------|-----------------------------------------------------------|------------------------------------------------------------------|
   | VilBert | balanced real | precision: 1, recall: 0.51, fscore: 0.68, vqascore: 0.94 | precision: 0.997, recall: 0.511, fscore: 0.676, vqa_score: 0.936 |
  
   

   调用ACL接口推理计算，性能参考下列数据。
   
   | 芯片型号 | Batch Size | 数据集        | 性能    |
   |----------|------------|---------------|---------|
   | 310P3    | 1          | balanced_real | 187 fps |
   | 310P3    | 4          | balanced_real | 223 fps |
   | 310P3    | 8          | balanced_real | 373 fps |
   | 310P3    | 16         | balanced_real | 488 fps |
   | 310P3    | 32         | balanced_real | 493 fps |
   | 310P3    | 64         | balanced_real | 428 fps |
   | 310      | 1          | balanced_real | 200 fps |
   | 310      | 4          | balanced_real | 228 fps |
   | 310      | 8          | balanced_real | 232 fps |
   | 310      | 16         | balanced_real | 236 fps |
   | 310      | 32         | balanced_real | 236 fps |
   | 310      | 64         | balanced_real | 236 fps |
