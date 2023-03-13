# Hubert模型离线推理指导
- [概述](#概述)
    - [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)

- [快速上手](#快速上手)

  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)

- [模型推理性能](#模型推理性能)

  ******

  

# 概述<a name="概述"></a>

HuBERT是一种学习自监督语音表征的新方法。通过在聚类和预测步骤之间交替进行，逐步提高其学习的离散表征。HuBERT可以从连续输入中学习声学和语言模型。HuBERT与SOTA方法在语音识别、语音生成、语音压缩的语音表征学习方面相匹配，甚至超过了 SOTA。

- 参考论文：[HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/pdf/2106.07447.pdf)

- 参考实现：

  ```
  url=https://github.com/facebookresearch/fairseq.git
  branch=main
  commit_id=5528b6a38224404d80b900609463fd6864fd115a
  ```


## 输入输出数据<a name="输入输出数据"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小         | 数据排布格式 |
  | ---- |------------| ----------------- | -------- |
  | source | FP32 | 1 x 580000 | NCHW       |


- 输出数据

  | 输出数据 | 大小            | 数据类型 | 数据排布格式 |
  |---------------| -------- |--------| ------------ |
  | result | 1812 x 1 x 32 | FLOAT32  | ND     |




# 推理环境准备<a name="推理环境准备"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------------ |---------| ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fpies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.11.0  | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 |         |                                                              |



# 快速上手<a name="快速上手"></a>

## 获取源码<a name="获取源码"></a>

1. 获取源码。

   ```
   git clone https://github.com/facebookresearch/fairseq.git -b main 
   cd fairseq
   git reset --hard 5528b6a38224404d80b900609463fd6864fd115a
   patch -p1 < ../hubert.patch
   cd ..
   mkdir data
   ```

2. 安装依赖。

   ```
   pip3.7.5 install -r requirements.txt
   ```

​		


## 准备数据集<a name="准备数据集"></a>

1. 获取原始数据集。

   用户自行获得[test-clean](https://www.openslr.org/resources/12/test-clean.tar.gz)数据集，解压到./data/

   运行hubert_data.sh脚本将数据集处理成tsv,ltr文件。
   
   ```
   bash hubert_data.sh
   ```

   解压后数据集目录结构：

    ```
    ├── data
    │     ├──LibriSpeech
    │        ├──test-clean
    │           ├──61
    │           	├──70968
    │                     ├──61-70968.trans.txt
    │                     ├──61-70968-0000.flac
    │                     ├──61-70968-0001.flac
    │                     ├──......
    │           	├──......	
    │           ├──......	
    │        ├──BOOKS.TXT
    │        ├──CHAPTERS.TXT
    │        ├──LICENSE.TXT
    │        ├──README.TXT
    │        ├──SPEAKERS.TXT
    │     ├──test-clean
    │        ├──train.wrd
    │        ├──train.tsv
    │        ├──train.ltr 
    ```

2. 数据预处理。

   将原始数据转化为二进制文件（.bin）。

   执行hubert_preprocess.py脚本，生成数据集预处理后的bin文件，存放在当前目录下的pre_data/test-clean文件夹中。模型权重获取方法见[模型推理](#模型推理)。

   ```
   mkdir -p ./pre_data/test-clean
   python3.7.5 hubert_preprocess.py --model_path ./hubert_large_ll60k_finetune_ls960.pt --datasets_tsv_path ./data/test-clean/train.tsv --datasets_ltr_path ./data/test-clean/train.ltr --pre_data_source_save_path ./pre_data/test-clean/source/ --pre_data_label_save_path ./pre_data/test-clean/label/
   ```
   
   - 参数说明
       - model_path：表示模型权重文件路径。
       - datasets_tsv_path：数据集tsv位置。
       - datasets_ltr_path：数据集ltr位置。
       - pre_data_source_save_path: source保存路径。
       - pre_data_label_save_path: label保存位置。
    ```
    ├── pre_data
    │   ├──test-clean
    |    ├──label 
    |      ├──label0.bin 
    |      ├──label1.bin
    |      ├──......  
    |    ├──source
    |      ├──source0.bin 
    |      ├──source1.bin 
    |      ├──...... 
    ```


## 模型推理<a name="模型推理"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pt转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从ModelZoo的源码包中获取Hubert权重文件[hubert_large_ll60k_finetune_ls960.pt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Hubert/PTH/hubert_large_ll60k_finetune_ls960.pt)。

   2. 导出onnx文件。

      1. 使用pth2onnx.py导出onnx文件。

         运行pth2onnx.py脚本。

         ```
         python3.7.5 pth2onnx.py --model_path hubert_large_ll60k_finetune_ls960.pt --onnx_path ./hubert.onnx
         ```

         获得hubert.onnx文件。
         - 参数说明：
             - --model_path：表示模型权重文件路径。
             - --onnx_path：生成的onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
         
         > **说明：** 
         该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
      
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
      
         使用atc将onnx模型转换为om模型文件，工具使用方法可以参考《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。生成转换batch size为1的om模型的命令如下，对于其他的batch size，可作相应的修改。
         
         ```
         atc --framework=5 --model=hubert.onnx --output=hubert --input_format=ND --input_shape="source:1,580000" --soc_version=Ascend${chip_name} --log=error
         ```
      
         - 参数说明：
           -   --framework：5代表ONNX模型。
           -   --model：为ONNX模型文件。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
 
         运行成功后生成hubert.om模型文件。
         
2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3.7.5 -m ais_bench --model hubert.om --batchsize 1 --input "./pre_data/test-clean/source/" --output "./out_data/test-clean/" --device 0
        ```
        - 参数说明：
            - --model: 需要进行推理的om离线模型文件。
            - --batchsize: 模型batchsize。
            - --input: 模型需要的输入，指定输入文件所在的目录即可。
            - --output: 推理结果保存目录。结果会自动创建”日期+时间“的子目录，保存输出结果。可以使用--output_dirname参数，输出结果将保存到子目录output_dirname下。
            - --device: 指定NPU运行设备。取值范围为[0,255]，默认值为0。

            推理后的输出默认在当前目录result下。

   3. 精度验证。

        调用hubert_postprocess.py脚本，可以获得error_rate数据，结果保存在./res_data/test-clean/error_rate.txt中。

        ```
        mkdir -p ./res_data/test-clean
        python3.7.5 hubert_postprocess.py --model_path ./hubert_large_ll60k_finetune_ls960.pt --source_json_path ./out_data/test-clean/2023_01_06-02_25_32_summary.json --label_bin_file_path ./pre_data/test-clean/label/ --res_file_path ./res_data/test-clean/
        ```

        - 参数说明：
            - --model_path: 表示模型权重文件路径。
            - --source_json_path: 表示离线推理输出所在的文件夹的json文件，路径为"./out_data/test-clean/*_summary.json (*号代表"日期+时间"的命名)。
            - --label_bin_file_path: 表示正确答案的文件路径。
            - --res_file_path: 表示输出精度数据所在的文件名。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7.5 -m ais_bench --model hubert.om --batchsize 1 --output ./result --loop 1000 --device 0
        ```

      - 参数说明：
        - --model: 需要进行推理的om模型。
        - --batchsize: 模型batchsize。不输入该值将自动推导。当前推理模块根据模型输入和文件输出自动进行组batch。参数传递的batchszie有且只用于结果吞吐率计算。请务必注意需要传入该值，以获取计算正确的吞吐率。
        - --output: 推理结果输出路径。默认会建立"日期+时间"的子文件夹保存输出结果。
        - --loop: 推理次数。默认值为1，取值范围为大于0的正整数。
        - --device: 指定NPU运行设备。取值范围为[0,255]，默认值为0。

   ​	

# 模型推理性能&精度<a name="模型推理性能&精度"></a>

调用ACL接口推理计算，精度和性能参考下列数据。

|   芯片型号   | Batch Size |    数据集     | 精度error_rate |  性能   |
|:--------:|:----------:|:----------:|:------------:|:-----:|
|  310P3   |     1      | test-clean |    2.136     | 3.158 |

注：只支持batchsize为1