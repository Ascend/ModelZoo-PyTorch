# Deepspeech2模型-推理指导
 

- [概述](#ZH-CN_TOPIC_0000001172161501)

  - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Deepspeech是百度推出的语音识别框架，系统采用了端对端的深度学习技术，也就是说，系统不需要人工设计组件对噪声、混响或扬声器波动进行建模，而是直接从语料中进行学习，并达到了较好的识别效果。



- 参考实现：

  ```
    url=https://github.com/SeanNaren/deepspeech.pytorch
    branch=master
    commit_id=075a69ae66aa284c5c5a954c6c15efe6d56898dd
  ```

  


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input1    | FLOAT32 | batchsize x 1 x 161 x 621| NCHW         |
  | input2    | INT32| batchsize x 1 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 311 x 29| FLOAT32  | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动


  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- |   ------------------------------------------------------------ |
  | 固件与驱动| 1.0.16（NPU驱动固件版本为5.1.RC2）                  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies)                                                        |
  |CANN|5.1.RC2|-                                                              |
  | Pytorch| 1.8.0 | -  |




# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码并安装。
   ```
   git clone https://github.com/SeanNaren/deepspeech.pytorch.git -b V3.0
   cd deepspeech.pytorch
   pip3 install -e .
   ```

2. 安装依赖。

    ```
    pip3 install -r requirement.txt
    ```
    > **说明：** 
    >torchaudio==0.8.0目前没有可以在arm环境下运行的包。

3. 在Deepspeech2目录下执行differences.patch文件，修改开源仓model.py文件。

    ```
    patch -p4 < differences.patch
    ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   
    ```
    cd deepspeech.pytorch
    python3 ./data/an4.py
    ```
    得到的数据结构为
    ```
    |——an4_test_manifest.json
    |——labels.json  
    |——an4_dataset
            |——val
            |——train
            |——test
    ```
    > **说明：** 
    >如下载不了，可在本地用vscode拉代码下载过后传到服务器。
    >获取原始数据前将源码移动到deepspeech.pytorch文件

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。
   执行deepspeech_preprocess.py脚本，完成预处理。

   ```
   python3 deepspeech2_preprocess.py --data_file ./data/an4_test_manifest.json --save_path ./data/an4_dataset/test --label_file ./labels.json
   ```
   - 参数说明:

      - --data_file：json文件路径。
   
      - --save_path：输出的二进制文件（.bin）所在路径。
   
      - --label_file：标签文件路径。

    > **说明：** 
    >在预处理前，修改an4_test_manifest.json中root_path参数，改为当前an4_dataset中test数据集的路径，方便进行数据预处理。
    >如果linux系统缺少sox，需要安装sox。

    运行成功后在./data/an4_dataset/test目录下生成供模型推理的bin文件。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

        获取权重文件：[an4_pretrained_v3.ckpt](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/DeepSpeech2/PTH/an4_pretrained_v3.ckpt)

   2. 导出onnx文件。

      1. 使用“an4_pretrained_v3.ckpt”导出onnx文件。
         运行“ckpt2onnx.py”脚本。
         

         ```
         python3 deepspeech2_ckpt2onnx.py --ckpt_path ./an4_pretrained_v3.ckpt --out_file deepspeech.onnx
         ```
    
         获得“deepspeech.onnx”文件。

   3. 使用ATC工具将ONNX模型转OM模型。

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
          atc --framework=5 --model=./deepspeech.onnx --input_format=NCHW --input_shape="spect:1,1,161,621;transcript:1" --output=deepspeech_bs1 --log=error --soc_version=${chip_name}
          ```

          - 参数说明：

              - --model：为ONNX模型文件。

              - --framework：5代表ONNX模型。

              - --output：输出的OM模型。

              - --input_format：输入数据的格式。

              - --input_shape：输入数据的shape。

              - --log：日志级别。

              - --soc_version：处理器型号。

        运行成功后生成“deepspeech_bs1.om”模型文件。



2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。


   2. 执行推理。

      ```
      python3 -m ais_bench --model ./deepspeech_bs1.om  --input ./data/an4_dataset/test/spect,./data/an4_dataset/test/sizes --output ./result --output_dir dumpout_bs1 --outfmt TXT --batchsize 1
      ```
    
      -   参数说明：
    
           - -- model：om文件路径。

           - -- input：输入的数据文件。

           - -- output：输出结果路径。
    
      推理后的输出默认在当前目录result下。
      并且会输出性能数据



    3. 精度验证。

        调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。
        执行deepspeech2_postprocess.py脚本，可以获得精度数据。
     
         ```
         python3 deepspeech2_postprocess.py --out_path ./result/dumpout_bs1 --info_path ./data/an4_dataset/test --label_file ./labels.json
         ```
         - 参数说明：

           - --out_path：生成推理结果所在路径。
    
           - --info_path：输出的二进制文件（.bin）所在路径。
    
           - --label_file：标签数据路径。

    4. 性能验证。

        可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
        python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
        ```
        
        - 参数说明：
           - --model：om模型的路径
           - --batchsize：数据集batch_size的大小


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

  精度如下列数据。

  |数据集 | 精度 |
  | ---------- | ---------- |
  |   an4        |       Average WER 9.573 Average CER 5.515     |


  调用ACL接口推理计算，性能参考下列数据。

  | 芯片型号 | Batch Size | 数据集   | 性能    |
  | -------- | ---------- | -------- | ------- |
  | 310P3    | 1          | an4 |  0.48  |
  | 310P3    | 4          | an4 |  1.93 |
  | 310P3    | 8          | an4 |  3.86 |
  | 310P3    | 16         | an4 |  7.7 |
  | 310P3    | 32         | an4 |  7.74 |
  | 310P3    | 64         | an4 |  7.48 |