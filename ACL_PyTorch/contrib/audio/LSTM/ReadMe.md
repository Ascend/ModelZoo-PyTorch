# LSTM模型-推理指导


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

长短期记忆网络（Long Short-Term Memory）是一种时间循环神经网络，是为了解决一般的RNN（循环神经网络）存在的长期依赖问题而专门设计出来的，主要用于解决长序列训练过程中的梯度消失和梯度爆炸问题。相比普通的RNN，LSTM能够在更长的序列中有更好的表现。


- 参考实现：

  ```
  url=https://gitee.com/ascend/modelzoo
  commit_id=718e303e705102860809894a623ae80e0103b7fd
  code_path=ACL_PyTorch\contrib\audio\LSTM
  model_name=LSTM
  ```
  



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                  | 数据排布格式 |
  | -------- | -------- | --------------------- | ------------ |
  | input    | FLOAT32  | batchsize x 390 x 243 | ND           |


- 输出数据

  | 输出数据 | 数据类型             | 大小    | 数据排布格式 |
  | -------- | -------------------- | ------- | ------------ |
  | output1  | 195 x batchsize x 41 | FLOAT32 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.8以上 | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

- 该模型需要以下依赖 

  **表 2**  依赖列表

  | 依赖名称        | 版本               |
  | --------------- | ------------------ |
  | ONNX            | 1.7.0              |
  | numpy           | 1.22.0以上         |
  | Pillow          | 7.2.0              |
  | onnxruntime-gpu | 1.7.0              |
  | kaldiio         | 2.17.2             |
  | kaldi           | 见下“安装依赖”部分 |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取数据处理脚本

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd /ModelZoo-PyTorch/PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/
   ```

   注：将当前目录下所有**文件夹**复制到“ModelZoo-PyTorch_1\ACL_PyTorch\contrib\audio\LSTM”目录下。

   将“ModelZoo-PyTorch_1\ACL_PyTorch\contrib\audio\LSTM”目录下的LSTM_postprocess_data.py，LSTM_preprocess_data.py，LSTM_pth2onnx.py 3个文件移动到steps文件夹中。
   
2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```
   
   下载kaldi工具包在ModelZoo-PyTorch/ACL_PyTorch/built-in/audio/LSTM下
   源码搭建kaldi工具包环境。以arm 64位环境为例说明，推荐安装至conda环境：

   ```
   git clone https://github.com/kaldi-asr/kaldi
   cd kaldi
   ```
   
   检查工具包所需依赖并安装缺少依赖。
   
   ```
   tools/extras/check_dependencies.sh
   ```
   
   
   
   根据检查结果和提示，安装缺少的依赖。安装完依赖再次检查工具包所需依赖是否都安装ok。
   
   若提示有缺少的依赖，根据提示进行安装。
   
   ```
   cd tools
   make -j 64
   ```
   
   安装依赖库成功之后安装第三方工具，Kaldi使用FST作为状态图的表现形式，安装方式如下：
   
   ```
   make openfst
   extras/install_irstlm.sh
   extras/install_openblas.sh
   ```
   
   ```
   输出：Installation of IRSTLM finished successfully
   输出：OpenBLAS is installed successfully
   ```
   
   配置源码：
   
   ```
   cd ../src/
   ./configure --shared
   输出"Kaldi has been successfully configured."
   ```
   
     编译安装：
   
   ```
   make -j clean depend
   make -j 64
      
   输出：Done
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持[timit语音包](链接：https://pan.baidu.com/s/1OhdCKQEIFCIq9iAFkoKRhQ?pwd=fkma 
   提取码：fkma [)的验证集。timit数据集与训练对齐，使用训练提供的语音数据包。需用户自行获取数据集，并将**数据集命名为data.zip**，并上传数据集data.zip至服务器ModelZoo-PyTorch/ACL_PyTorch/contrib/audio/目录中。数据集结构如下。

   ```
   ├── DOC
   ├── README.DOC
   ├── TEST
   └── TRAIN
   ```

2. 数据预处理。

   数据预处理原始数据（audio数据）转化为二进制文件（.bin）。
1.解压数据集
   ```
   unzip data.zip
   cd LSTM
   ```
   
   此处data.zip为数据集压缩包。
   
   2.创建data文件夹
   
   ```
    mkdir data
   ```
   
   3.执行prepare_data.sh脚本。
   
   ```
   chmod +x local/timit_data_prep.sh
   chmod +x steps/make_feat.sh
   bash prepare_data.sh
   ```
   
   若出现文件不存在，请检查是否将/ModelZoo-PyTorch/PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/中的文件复制到当前目录。
   
   如果出现\r报错：
   
   ```
   vi prepare_data.sh 
   Esc进入命令行运行模式
   :set ff=unix
   :wq 
   ```
   
   执行prepare_data.sh脚本之后，在当前目录下会生成tmp文件夹和在data文件夹下生成dev,test,train三个数据集文件夹。使用此目录下的dev数据集进行验证。
   
    4.执行LSTM_preprocess_data.py脚本
   
   ```
   python3 ./steps/LSTM_preprocess_data.py --conf=./conf/ctc_config.yaml --batchsize=16
   ```
   
   + 参数说明：
     + ./conf/ctc_config.yaml：配置文件路径。

​		不同batchsize模型需要修改--batchsize参数，生成不同的数据。执行后在当前目录下生成lstm_bin文件夹。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从源码包中获取权重文件[ctc_best_model.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/LSTM/PTH/ctc_best_model.pth)。
    在LSTM目录下创建checkpoint/ctc_fbank_cnn/目录并将权重文件移到到该目录下。
       
       ```
       mkdir -p checkpoint/ctc_fbank_cnn
       mv ./ctc_best_model.pth ./checkpoint/ctc_fbank_cnn/
       ```

   2. 执行steps/LSTM_pth2onnx.py脚本将.pth文件转换为.onnx文件

      ```
      python3 ./steps/LSTM_pth2onnx.py --batchsize=16
      ```

      获得lstm_ctc_16batch.onnx文件。

      注：若有warnings.warn("Exporting a model to ONNX with a batch_size other than 1, " +  无需理会。

      不同batchsize模型需要修改--batchsize参数。

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
         atc --input_format=ND --framework=5 --model=lstm_ctc_16batch.onnx --input_shape="actual_input_1:16,390,243" --output=lstm_ctc_16batch --log=info --soc_version=Ascend{chip_name}
         ```
         
         - 参数说明：
         
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           
         
         其中，不同batchsize模型除需要修改输入输出模型名称外，还需将输入改为"actual_input_1:{batchsize},390,243"
         运行成功后生成lstm_ctc_16batch.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        创建文件夹lcmout，并在此文件夹下分别创建bs1 - bs64 6个子文件夹。

        ```
        python3 -m ais_bench --model ./lstm_ctc_16batch.om --input ./lstm_bin/ --output ./lcmout/bs16/ --outfmt NPY --batchsize 16
        ```

        -   参数说明：

             -   --model：om模型。
             -   --input：预处理数据集路径。
             -   --output：推理结果所在路径。
             -   --outfmt：推理结果文件格式。
             -   --batchsize：不同的batchsize。

        不同batchsize模型需要修改--batchsize参数和相应的模型、输出文件夹名称

        推理后的输出默认在当前目录result下。


   3. 精度验证。

      执行steps/LSTM_postprocess_data.py脚本进行数据后处理。

      ```
      python3 ./steps/LSTM_postprocess_data.py --conf=./conf/ctc_config.yaml --npu_path=./lcmout/bs16/xxxx --batchsize=16
      ```

      - 参数说明：

        - --conf：模型配置文件
        - --npu_path：推理结果目录，此处需要改为文件夹下最新结果路径（以日期时间命名的文件夹）。

      执行后处理脚本之后，精度数据由WER 与CER给出，分别为字母错误率与单词错误率。示例如下：

      ```
      Character error rate on test set: 13.5877
      Word error rate on test set: 18.9075
      ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度(character) | 精度(word) | 性能    |
| -------- | ---------- | ------ | --------------- | ---------- | ------- |
| 310P3    | 1          | TIMIT  | 13.5477         | 18.8244    | 5.3296  |
| 310P3    | 4          | TIMIT  | 13.5477         | 18.8244    | 21.1861 |
| 310P3    | 8          | TIMIT  | 13.5477         | 18.8244    | 41.741  |
| 310P3    | 16         | TIMIT  | 13.5477         | 18.8244    | 82.4728 |
| 310P3    | 32         | TIMIT  | 13.5477         | 18.8244    | 83.0379 |
| 310P3    | 64         | TIMIT  | 13.5477         | 18.8244    | 83.4891 |