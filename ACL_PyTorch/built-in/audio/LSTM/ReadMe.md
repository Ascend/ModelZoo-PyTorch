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

LSTM是一种特殊的RNN模型，与普通RNN相比，LSTM可以更好地解决长序列训练过程中的梯度消失和梯度爆炸问题。


- 参考实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  commit_id=8ed54e7d0fc9b632e1e3b9420bed96ee2c7fa1e3
  code_path=ModelZoo-PyTorch/PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch
  model_name=LSTM
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32 | batchsize x 390 x 243 | ND         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | 195 x batchsize x 41 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本      | 环境准备指导                                                 |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.4  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.3.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd ModelZoo-PyTorch
   cp -r PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/NPU/1p/* ACL_PyTorch/built-in/audio/LSTM/
   cd ACL_PyTorch/built-in/audio/LSTM
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

3. 安装kaldi工具包，以arm 64位环境为例说明，推荐安装至conda环境：
   
   1. 下载kaldi工具包
      源码搭建kaldi工具包环境。
      ```
      git clone https://github.com/kaldi-asr/kaldi
      cd kaldi
      ```

   2. 检查工具包所需依赖并安装缺少依赖。
      ```
      chmod +x tools/extras/check_dependencies.sh
      tools/extras/check_dependencies.sh
      ```
      根据检查结果和提示，安装缺少的依赖，跳过安装MKL，用OpenBLAS替代。安装完依赖再次检查工具包所需依赖是否都安装成功。
      > **说明：**
      > 源码中使用的python2.7版本，如果系统python版本与该版本不同，可使用系统默认python，在目录kaldi/python/下创建空文件.use_default_python。其他安装问题可参见kaldi官方安装教程.
      
   3. 编译。
      ```
      cd tools
      make -j 64
      ```
      
      ```
      输出：All done OK.
      ```

   4. 安装依赖库成功之后安装第三方工具，安装方式如下：
      ```
      chmod +x extras/install_openblas.sh
      extras/install_openblas.sh
      ```

      ```
      输出：OpenBLAS is installed successfully
      ```

   5. 配置源码。
      ```
      cd ../src/
      ./configure --shared --mathlib=OPENBLAS
      ```
      
      ```
      输出：Kaldi has been successfully configured.
      ```

   6. 编译安装。
      ```
      chmod +x base/get_version.sh
      make -j clean depend
      make -j 64
      ```

      ```
      输出：Done
      ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持timit语音包的验证集。timit数据集与训练对齐，使用训练提供的语音数据包。需用户自行获取数据集，并将数据集命名为data.zip，并上传数据集data.zip至服务器模型源码包所在目录。数据集结构如下。

   ```
   ├── DOC
   ├── README.DOC
   ├── TEST
   └── TRAIN
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   模型输入数据为二进制格式。将原始数据（audio数据）转化为二进制文件（.bin）。
   1. 解压数据集。
      ```
      unzip data.zip
      ```
   2. 修改path.sh里第一行代码如下。
      ```
      KALDI_ROOT=./kaldi
      ```
   3. 执行prepare_data.sh脚本。
      ```
      chmod +x local/timit_data_prep.sh
      chmod +x steps/make_feat.sh
      bash prepare_data.sh
      ```
      在当前目录下会生成tmp文件夹和在data文件夹下生成dev,test,train三个数据集文件夹。
   4. 移动LSTM_preprocess_data.py至steps目录下。
      ```
      mv LSTM_preprocess_data.py ./steps/
      ```
   5. 修改./conf/ctc_config.yaml文件内容。
      ```
      #[test]
      test_scp_path: 'data/dev/fbank.scp'
      test_lab_path: 'data/dev/phn_text'
      decode_type: "Greedy"
      beam_width: 10
      lm_alpha: 0.1
      lm_path: 'data/lm_phone_bg.arpa'
      ```
      使用data文件夹下的dev数据集进行验证。
   6. 执行LSTM_preprocess_data.py脚本。
      ```
      python3 ./steps/LSTM_preprocess_data.py --conf=./conf/ctc_config.yaml --batchsize=${batch_size}
      ```
      
      - 参数说明：
        - --conf：模型配置文件。
        - --batchsize：推理输出batch大小。
      获得lstm_bin_bs${batch_size}二进制文件夹。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      ctc_best_model.pth权重文件[下载链接](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/LSTM/PTH/ctc_best_model.pth)。在当前目录下创建checkpoint/ctc_fbank_cnn/目录并将权重文件移到到该目录下。
      ```
      mkdir -p checkpoint/ctc_fbank_cnn
      mv ./ctc_best_model.pth ./checkpoint/ctc_fbank_cnn/
      ```

   2. 导出onnx文件。

      移动LSTM_pth2onnx.py至steps目录下，使用LSTM_pth2onnx.py导出onnx文件。注：如果torch版本高于1.8，需要将models/model_ctc.py文件中import torch_npu注释。

      ```
      mv LSTM_pth2onnx.py ./steps/
      python3 ./steps/LSTM_pth2onnx.py
      ```

      获得lstm_ctc.onnx文件。

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
         atc --input_format=ND --framework=5 --model=lstm_ctc.onnx --input_shape="actual_input_1:${batch_size},390,243" --output=lstm_ctc_bs${batch_size} --log=info --soc_version=Ascend${chip_name} 
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***lstm_ctc_bs${batch_size}.om***</u>模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3 -m ais_bench --model=lstm_ctc_bs${batch_size}.om --input=lstm_bin_bs${batch_size} --output=result --output_dirname=bs${batch_size} --outfmt=NPY
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入数据目录。
             -   --output：推理结果输出路径。
             -   --output_dirname：推理结果输出目录。
             -   --outfmt：推理结果输出格式。

   3. 精度验证。

      移动LSTM_postprocess_data.py脚本至steps目录下，执行LSTM_postprocess_data.py脚本进行数据后处理。

      ```
      mv LSTM_postprocess_data.py ./steps
      python3 ./steps/LSTM_postprocess_data.py --conf=./conf/ctc_config.yaml --npu_path=./result/bs${batch_size} --batchsize=${batch_size}
      ```

      - 参数说明：
        - --conf：模型配置文件。
        - --npu_path：推理输出目录。
        - --batchsize：推理输出batch大小。

   4. 性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=lstm_ctc_bs${batch_size}.om --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om文件路径。
        - --batchsize：batch大小。

    
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度                      | 性能          |
| --------- |------------| ---------- |-------------------------|-------------|
|   Ascend310P3        | 1          |   TIMIT         | CER:13.5491/WER:18.8003 | 5.3457 fps  |
|   Ascend310P3        | 4          |   TIMIT         | CER:13.5370/WER:18.7877 | 21.3256 fps |
|   Ascend310P3        | 8          |   TIMIT         | CER:13.5156/WER:18.7729 | 42.0914 fps |
|   Ascend310P3        | 16         |   TIMIT         | CER:13.5502/WER:18.8313 | 82.6230 fps |
|   Ascend310P3        | 32         |   TIMIT         | CER:13.5502/WER:18.8313 | 83.1510 fps |
|   Ascend310P3        | 64         |   TIMIT         | CER:13.5502/WER:18.8313 | 83.5096 fps |