# {ELMo}模型-推理指导


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

ELMo模型是用于训练得到单词词向量的，不同于以往常用的word2vec（CBOW、SkipGram、Hierarchical Softmax）通过大规模语料对每一个单词训练出固定的词向量，ELMo可以通过不同的语句对同一个单词训练得到不同的词向量，有效区分出同一个单词在不同语境下表示的不同含义（例如：apple可以表示苹果，也可以表示iphone）。

- 参考实现：

  ```
  url=https://github.com/allenai/allennlp/blob/main/allennlp/modules/elmo.py
  model_name=elmo
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | INT32 | 1 x 8 x 50 | NWD         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output  | 1 x 8 x 1024 | FLOAT32  | NWD           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.9.1   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   ```
   git clone https://github.com/allenai/allennlp.git
   cd allennlp
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   该模型使用`1 Billion Word Language Model Benchmark`数据集进行推理，[下戴链接](https://www.statmt.org/lm-benchmark/),该数据集包含训练集和测试集，下载后解压在当前目录下。文件夹名称为`1-billion-word-language-modeling-benchmark-r13output`。

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行elmo_preprocess.py脚本，完成预处理。

   ```
   python3 elmo_preprocess.py \
       --file_path ./1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/ \
       --save_path ./data.txt \
       --bin_path ./bin_path \
       --file_num 50 \
       --word_len 8
   ```

   - 参数说明：
     
     - --file_path：测试数据集路径
     - --save_path：中间文件路径
     - --bin_path：预处理数据集路径
     - --file_num：测试数据集数量
     - --word_len：取句子长度
   
   运行成功之后产生的data.txt文件为从原始测试集中筛选的符合模型输入的数据（即句子长度小于等于8的句子）；bin_path下为处理好的每个句子的二进制文件


## 模型推理<a name="section741711594517"></a>

1. 获取权重文件

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。
   
   下载链接：https://allenai.org/allennlp/software/elmo

   选择Pre-trained ELMo Models中的Original模型，并下载对应的`weights`和`options`放在当前目录下，文件名称分别为`elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5`，`elmo_2x4096_512_2048cnn_2xhighway_options.json`。
       
2. 模型修改
   
   修改环境allennlp库下elmo.py文件，示例路径：/root/anaconda3/envs/elmo/lib/python3.7/site-packages/allennlp/modules/elmo.py

   将609行的torch.chunk函数修改为torch.split函数，原因：onnx opset version11 不再支持chunk函数，可以换成支持的split函数。

3. 模型转换

   1. 导出onnx文件。

      1. 使用elmo_pth2onnx.py导出onnx文件。

         运行elmo_pth2onnx.py脚本。

         ```
         python3 elmo_pth2onnx.py --output_file elmo.onnx --word_len 8
         ```

      - 参数说明:

        - output_file：导出onnx文件路径
        - word_len：模型输出句子长度
  
         获得elmo.onnx文件。

      1. 优化ONNX文件。

         ```
         python3 -m onnxsim elmo.onnx elmo_sim.onnx
         ```

         获得elmo_sim.onnx文件。

   2. 使用ATC工具将ONNX模型转OM模型。

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
         atc --framework=5 --model=elmo_sim.onnx --output=elmo_sim --input_format=ND --input_shape="input:1,8,50" --log=error --soc_version=Ascend310P3
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成elmo_sim.om模型文件。



4. 开始推理验证。

   1.  使用ais-infer工具进行推理。

       ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2.  执行推理。

       ```
       python3 ${path_to_ais-infer}/ais_infer.py \
           --model ./elmo_sim.om \
           --input ./bin_path \
           --output ./ 
       ```

      -   参数说明：

           -   --model：模型路径
           -   --input：数据集文件夹路径
           -   --output：输出路径

      `${path_to_ais-infer}`为ais_infer.py脚本的存放路径，推理完成后在当前工作目录生成推理结果，命名格式为`xxxx_xx_xx-xx_xx_xx`(`年_月_日-时_分_秒`)，如`2022_08_18-06_55_19`。

      >**说明：** 
      >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   3.  精度对比。

       1. 调用脚本将onnx推理结果和om推理结果比对，计算余弦相似度，相似度大于99%即为达标。

         ```
         python3 elmo_postprocess.py \
             --onnx_model elmo_sim.onnx  \
             --onnx_input "bin_path/" \
             --om_out ${output_path}
         ```

       -  参数说明：

          - --onnx_model：onnx模型路径。
          - --onnx_input：onnx模型输入路径。
          - --om_out：om模型推理结果。
       
       ${output_path}为推理结果的保存路径，与onnx推理结果比对余弦相似度为99.99%，精度达标。


   4.  性能对比。

       可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

       ```
       python3 ${path_to_ais-infer}/ais_infer.py --model ./elmo_sim.om --loop=20 
       ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|       Ascend 310P3    |     1             |     1 Billion Word Language Model Benchmark R13 Output       |     cosine similarity:99.99%       |       41.78fps          |