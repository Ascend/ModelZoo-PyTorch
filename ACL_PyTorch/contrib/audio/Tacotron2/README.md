# tacotron2模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section183221994400)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)






# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Tacotron 模型是一个直接从文本合成语音的神经网络架构。系统由两部分构成，一个循环seq2seq结构的特征预测网络，把字符向量映射为梅尔声谱图，后面再接一个WaveNet模型的修订版，把梅尔声谱图合成为时域波形。

- 参考实现：

  ```
  url=https://github.com/NVIDIA/DeepLearningExamples.git 
  branch=master
  commit_id=9a6c5241d76de232bc221825f958284dc84e6e35
  model_name=tacotron2
  ```

  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- encoder输入数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | sequences | batchsize x text_seq | INT64 | ND |
  | sequence_lengths | batchsize | INT32 | ND |


- encoder输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | memory | batchsize x mem_seq x 512| FLOAT32  | ND |
  | processed_memory | batchsize x mem_seq x 128 | FLOAT32  | ND |
  | lens  | batchsize | FLOAT32  | INT32 |

- decoder_iter输入数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | decoder_input | batchsize x 80 | FLOAT32 | ND |
  | attention_hidden | batchsize x 1024 | FLOAT32 | ND |
  | attention_cell | batchsize x 1024 | FLOAT32 | ND |
  | decoder_hidden | batchsize x 1024 | FLOAT32 | ND |
  | decoder_cell | batchsize x 1024 | FLOAT32 | ND |
  | attention_weights | batchsize x seq_len | FLOAT32 | ND |
  | attention_weights_cum | batchsize x seq_len | FLOAT32 | ND |
  | attention_context | batchsize x 512 | FLOAT32 | ND |
  | memory | batchsize x seq_len x 512 | FLOAT32 | ND |
  | processed_memory | batchsize x seq_len x 128 | FLOAT32 | ND |
  | mask | batchsize x seq_len | BOOL | ND |


- decoder_iter输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | decoder_output | batchsize x 80 | FLOAT32 | ND |
  | gate_prediction | batchsize x 1 | FLOAT32 | ND |
  | out_attention_hidden | batchsize x 1024 | FLOAT32 | ND |
  | out_attention_cell | batchsize x 1024 | FLOAT32 | ND |
  | out_decoder_hidden| batchsize x 1024 | FLOAT32 | ND |
  | out_decoder_cell | batchsize x 1024 | FLOAT32 | ND |
  | out_decoder_weights | batchsize x seq_len | FLOAT32 | ND |
  | out_decoder_weights_cum | batchsize x seq_len | FLOAT32 | ND |
  | out_attention_context | batchsize x 512 | FLOAT32 | ND |


- postnet输入数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | mel_outputs | batchsize x 80 x mel_seq | FLOAT32 | ND |


- postnet输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | mel_outputs_postnet  | batchsize x 80 x mel_seq | FLOAT32  | ND           |




# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>
安装依赖。

   ```
   pip3 install -r requirements.txt
   pip3 install git+https://github.com/NVIDIA/dllogger@v0.1.0#egg=dllogger

   ```
   > **须知：** 
   > dllogger容易安装失败，可通过下载《[源码](https://github.com/NVIDIA/dllogger)》，解压后进入dllogger目录，执行python3 setup.py install。

   om_gener安装

   ```
   git clone https://gitee.com/peng-ao/om_gener.git
   cd om_gener
   pip3 install .
   ```

## 获取源码<a name="section183221994400"></a>
在工作目录下执行下述命令获取源码并切换到相应路径。

   ```
   git clone https://github.com/NVIDIA/DeepLearningExamples.git   
   cd DeepLearningExamples
   git reset --hard 9a6c5241d76de232bc221825f958284dc84e6e35
   cd PyTorch/SpeechSynthesis/Tacotron2
   tacotron_path=$(pwd)
   mkdir output
   mkdir checkpoints
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

    本模型支持LJ Speech 13100条文本数据集，包括12500条训练数据集，100条校验数据集和500条测试数据集。

2. 数据预处理。

   参考《[开源代码](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2/filelists)》中处理后的数据集，用于推理



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   本模型基于开源框架PyTorch训练的Tacotron2进行模型转换。使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

    1. 在checkpoints目录下获取权重文件。

        ```
        cd checkpoints
        wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_fp32/versions/19.09.0/zip -O waveglow_ckpt_fp32_19.09.0.zip
        unzip waveglow_ckpt_fp32_19.09.0.zip
        wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_fp32/versions/19.09.0/zip -O tacotron2_pyt_ckpt_fp32_19.09.0.zip
        unzip tacotron2_pyt_ckpt_fp32_19.09.0.zip
        ```

        > **须知：** 
        > waveglow作为辅助模型，用于验证tacotron2模型的精度。tacotron2输出频谱文件，通过waveglow得到音频文件。
   2. 导出onnx文件。
      1. 使用ModelZoo获取的源码包中的文件替换“DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2”目录下的文件。
         ```
         git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
         cd ModelZoo-PyTorch/ACL_PyTorch/contrib/audio/Tacotron2
         cp acl_net.py addweight.py om_infer_acl.py onnx_infer.py data_process.py atc_static.sh onnxsim.sh $tacotron_path
         cp get_out_node.py get_out_type.py $tacotron_path/output
         cp convert_tacotron22onnx.py convert_waveglow2onnx.py $tacotron_path/tensorrt 
         ```
      2. 需侵入式修改onnx文件，修改/usr/local/python3.7.5/lib/python3.7/site-packages/onnx/init.py， 在load_model函数中函数首添加load_external_data=False，并且在/usr/local/python3.7.5/lib/python3.7/site-packages/onnx/checker.py中的check_model中C.check_model(protobuf_string)注释掉，以上文件路径可能与描述不同，在自己安装的onnx路径下查找即可
    
      3. 使用pth导出onnx。

            1.运行convert_tacotron22onnx.py，并备份权重。

            ```
            cd $tacotron_path 
            python3.7 tensorrt/convert_tacotron22onnx.py --tacotron2 ./checkpoints/nvidia_tacotron2pyt_fp32_20190427 -o output/ --fp32 --batch_size=4 --iter=1
            cd output
            mkdir iter
            mv *.weight 3* iter/
            ```
         - 参数说明：

          - 第一个参数代表权重路径。
          - 第二个参数代表输出路径。
          - 第三个参数代表参数是fp32。
          - 第四个参数代表模型batch。
          - 第五个参数代表层数。
         > **说明：** 
         >注意使用不同版本的onnx生成的权重节点名称可能不同，请自行更改上述生成的权重名称。
         
            2.生成100层或其它层数叠加的decoder模型。

            ```
            cd $tacotron_path
            python3.7 tensorrt/convert_tacotron22onnx.py --tacotron2 ./checkpoints/nvidia_tacotron2pyt_fp32_20190427 -o output/ --fp32 --batch_size=4 --iter=100
            cd output
            rm -rf 2*
            ```
         - 参数说明：

          - 第一个参数代表权重路径。
          - 第二个参数代表输出路径。
          - 第三个参数代表参数是fp32。
          - 第四个参数代表模型batch。
          - 第五个参数代表层数。
           > **说明：** 
           >注意该步骤生成的权重占用磁盘空间约7GB，请注意预留空间。

            3.生成waveglow模型。
            ```
            cd $tacotron_path
            python3.7 tensorrt/convert_waveglow2onnx.py --waveglow ./checkpoints/nvidia_waveglowpyt_fp32_20190427 --config-file config.json -o output/ --fp32
            ```
            - 参数说明：

          - 第一个参数代表需要共享的权重路径。
          - 第二个参数代表配置文件。
          - 第三个参数代表输出路径。
          - 第四个参数参数是fp32。
          
        获得encoder.onnx, decoder_iter.onnx, postnet.onnx文件。

2. 优化ONNX文件。

    1. 执行命令增加工具可执行权限。

        ```
        chmod +x onnxsim.sh
        ```

    2. 修改decoder_iter模型，将权重共享并加载到onnx中, 并进行优化。

        ```
        python3 addweight.py $tacotron_path/output/iter $tacotron_path/output/decoder_iter.onnx $tacotron_path/output/decoder_iter_weight.onnx
        ```

        - 参数说明：

          - 第一个参数代表需要共享的权重路径。
          - 第二个参数代表共享权重前的onnx文件。
          - 第三个参数代表共享权重后的onnx文件。

        ```
        ./onnxsim.sh $tacotron_path/output/decoder_iter_weight.onnx $tacotron_path/output/decoder_sim_100.onnx 4 128
        ```

        - 参数说明：

          - 第一个参数代表需要优化的onnx文件。
          - 第二个参数代表优化后的onnx文件。
          - 第三个参数代表batchsize。
          - 第三个参数代表seq_len。

       获得decoder_sim_100.onnx文件。

3. 使用ATC工具将ONNX模型转OM模型。

    1. 配置环境变量。

        ```
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        export ASCEND_GLOBAL_LOG_LEVEL=3
        /usr/local/Ascend/driver/tools/msnpureport -g error -d 0
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
        bash atc_static.sh Ascend${chip_name} 4 128# Ascend310P3
        ```

        - 参数说明：

          - 第一个参数代表芯片类型。
          - 第二个参数代表batchsize。
          - 第三个参数代表seq_len。


        运行成功后生成<u>***encoder_static.om***, ***decoder_static.om***, ***postnet_static.om***</u>模型文件。

4. 开始推理验证。

    a.  使用om_infer_acl.py进行推理, 该文件调用aclruntime的后端封装的python的whl包进行推理。

    ```
    python3.7 om_infer_acl.py -i filelists/ljs_audio_text_test_filelist.txt -bs 4 -max_inputlen 128 -max_decode_iter 20 --device_id 0
    ```
    - 参数说明：

    - 第一个参数代表数据输入路径。
    - 第二个参数代表模型batch。
    - 第三个参数代表最大输入长度。
    - 第四个参数代表模型decode层数。
    - 第五个参数代表芯片序号。
   运行成功后生成result.txt文件，记录性能。
   
    b.  精度验证。

    此模型特殊，模型精度依靠人主观和原始的txt文本比对。



# 模型推理性能<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用aclruntime推理计算，性能参考下列数据。

| 模型              | 310性能   | 310P性能   | T4性能     | 310P/310 | 310P/T4 |
|-----------------|---------|----------|----------|----------|---------|
| Tacotron2(bs1)  | 1611.32 | 2942.48  | 606.36   | 1.82     | 4.85    |
| Tacotron2(bs4)  | 5762.64 | 9808.69  | 2431.74  | 1.70     | 4.03    |
| Tacotron2(bs8)  | 9959.2  | 18891.84 | 4312.98  | 1.89     | 4.38    |
| Tacotron2(bs16) | 15664.52| 33508.43 | 9040.19  | 2.13     | 3.71    |
| Tacotron2(bs32) | 9059    | 8965.99  | 11328.52 | 0.99     | 0.79    |
| 最优bs          | 15664.52| 33508.43 | 11328.52 | 2.13     | 2.95    |
> **须知：** 
>tacotron2_items_per_sec作为本模型性能评价指标。
