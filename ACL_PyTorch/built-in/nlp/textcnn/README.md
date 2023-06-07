# Textcnn模型PyTorch离线推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)


- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Textcnn是NLP模型，主要用于文本分析，使用预训练的word2vec初始化会使textcnn文本分析的效果更好，而非动态的比静态的效果好一些。总的来看，使用预训练的word2vec初始化的TextCNN，效果更好。


- 参考实现：

  ```
  git clone https://gitee.com/zhang_kaiqi/ascend-textcnn.git
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32 | batchsize x 32 | ND         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | --------| -------- | -------- | ------------ |
  | output  | FLOAT32  | batchsize x 10 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | [CANN推理架构准备](https://www/hiascend.com/software/cann/commercial) |
  | Python                                                       | 3.7.5   | 创建anaconda环境时指定python版本即可，conda create -n ${your_env_name} python==3.7.5 |
  | PyTorch                                                      | 1.11.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/zhang_kaiqi/ascend-textcnn.git
   cd  ascend-textcnn
   git checkout 7cd94c509dc3f615a5d8f4b3816e43ad837a649e
   cd ..
   git clone https://gitee.com/ascend/msadvisor.git
   ```

2. 安装依赖，由于有改图部分所以需要安装auto-optimizer

   ```
   cd msadvisor/auto-optimizer
   python3 -m pip install --upgrade pip
   python3 -m pip install wheel
   python3 -m pip install .
   cd ../..
   pip3.7 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

    本模型支持前文下载仓库ascend-textcnn内的Chinese-Text-Classification-Pytorch/THUCNews
    

2. 数据预处理

   1. 然后将原始数据集转换为模型输入的数据，执行TextCNN_preprocess.py脚本，完成预处理。
      ```
      cd ascend-textcnn
      mv ../TextCNN_preprocess.py ./
      mv ../TextCNN_pth2onnx.py ./
      mv ../TextCNN_postprocess.py ./
      mv ../gen_dataset_info.py ./
      python3 TextCNN_preprocess.py --save_folder bin
      python3 gen_dataset_info.py bin info
      ```

      - 参数说明：

         -   --save_folder：保存二进制数据集的路径。
         -   bin：读取二进制数据集的路径。
         -   info：生成的信息文件

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件，wget获取的权重如果在导出onnx文件时报错的话，请使用链接在网页中获取权重，并放于ascend-textcnn目录下

      ```
      wget https://gitee.com/hex5b25/ascend-textcnn/raw/master/Chinese-Text-Classification-Pytorch/THUCNews/saved_dict/TextCNN_9045_seed460473.pth
      ```

   2. 导出onnx文件。

      1. 使用TextCNN_pth2onnx.py导出onnx文件。

         运行ascend-textcnn目录下的TextCNN_pth2onnx.py脚本。

         ```
         python3 TextCNN_pth2onnx.py --weight_path ./TextCNN_9045_seed460473.pth --onnx_path ./dy_textcnn.onnx
         ```
         获得dy_textcnn.onnx文件。

         - 参数说明：

            -   --weight_path：权重路径。
            -   --onnx_path：生成onnx模型的路径。

      2. 简化onnx文件

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         cd ..
         mkdir ./onnx_sim_dir
         python3 -m onnxsim --input-shape="sentence:64,32" ./ascend_textcnn/dy_textcnn.onnx ./onnx_sim_dir/textcnn_64bs_sim.onnx
         ```

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
         mkdir ./mg_onnx_dir
         python3 ./fix_onnx.py 64
         mkdir ./mg_om_dir
         atc --model=mg_onnx_dir/textcnn_64bs_mg.onnx --framework=5 --output=mg_om_dir/textcnn_64bs_mg --output_type=FP16 --soc_version=Ascend${chip_name} --enable_small_channel=1
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --output：输出的OM模型。
           -   --soc\_version：处理器型号。
           -   --log：日志级别。
           -   --output_type：输出数据类型

           由于bs=1的情况不需要进行改图，故可以直接通过atc命令：atc --model=onnx_sim_dir/textcnn_1bs_sim.onnx --framework=5 --output=mg_om_dir/textcnn_1bs_mg --output_type=FP16 --soc_version=Ascend${chip_name} --enable_small_channel=1

2. 开始推理验证。

   1. 使用ais_bench工具进行推理。

      ais_bench工具获取及使用方式请点击查看[ais_bench推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)

   2. 执行推理。

        ```
        mkdir ./output_data
        python3.7 -m ais_bench --model mg_om_dir/textcnn_64bs.om --input ./ascend-textcnn/bin --output ./output_data --device 0  
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --input：数据预处理后保存文件的路径。
             -   --output：输出文件夹路径。
             -   --device：NPU的ID，默认填0。

        推理输出结果在output_data生成{20xx_xx_xx-xx_xx_xx}文件夹。



   4. 精度验证。

      调用TextCNN_postprocess.py脚本，可以获得精度accuracy数据，输入指令后请稍等片刻

      ```
      cd ascend-textcnn
      python3 TextCNN_postprocess.py ../output_data/{20xx_xx_xx-xx_xx_xx} > result_bs1.json
      ```

      - 参数说明：

        -   ../output_data/{20xx_xx_xx-xx_xx_xx}：为生成推理结果所在相对路径。  
        -   result_bs1.json：为保存精度数值的文件。


   5. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7 -m ais_bench --model mg_om_dir/textcnn_64bs_mg.om
        ```

      - 参数说明：
        - --model：om模型的路径



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据，官网精度为91.22。

| 芯片型号 | Batch Size | 数据集|  精度TOP1 | 精度TOP5 | 性能|
| --------- | ----| ----------| ------     |---------|---------|
| 310P3 |  1       | ImageNet |   90.47     |   99.35  |   5140.8      |
| 310P3 |  4       | ImageNet |   90.47     |   99.35  |    16167.36      |
| 310P3 |  8       | ImageNet |   90.47     |   99.35  |  21826.91     |
| 310P3 |  16       | ImageNet |   90.47     |   99.35  |   25861.78      |
| 310P3 |  32       | ImageNet |   90.47     |   99.35  |   27918.85      |
| 310P3 |  64       | ImageNet |   90.47     |   99.35  |   29237.18      |