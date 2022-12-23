# CNN_Transformer模型-推理指导


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

CNN_Transformer是学习用于解决自监督任务的基本语音单元。对模型进行训练，以预测语音中用于拟合目标的语音单元，同时学习到用于任务学习的语音建模单元应该是什么。模型首先使用多层卷积神经网络处理语音音频的原始波形，以获得每个25ms的潜在音频表示。这些表征向量被喂到量化器（quantizer）和transformer中。量化器从学习的单元清单（inventory of learned units）中选择一个语音单元作为潜在音频表征向量。大约一半的音频表示在被馈送到transformer之前被隐蔽掉（masked）。transformer从整个音频序列中添加信息，输出用于计算loss function。


- 参考实现：

  ```
  url=https://github.com/facebookresearch/fairseq/
  tag=v0.12.3
  code_path=/examples/wav2vec/
  model_name=wav2vec 2.0
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32  |  batchsize x data_len     | ND           |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | output   | FLOAT32  | batchsize x data_len x 32 | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.9.1   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/facebookresearch/fairseq.git        # 克隆仓库的代码
   cd fairseq/examples/wav2vec              # 切换到模型的代码仓目录
   git checkout main         # 切换到对应分支
   git reset --hard 0272196aa803ecc94a8bffa6132a8c64fd7de286      # 代码设置到对应的commit_id（可选）
   cd ${CNN_Transformer_Path}
   ```
   `${CNN_Transformer_Path}`为模型的根目录下
2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 数据预处理，将原始数据集转换为模型输入的数据。

   运行`CNN_Transformer_Preprocess.py`脚本，会自动在线下载所需的分词器模型、Librispeech数据集（下载过程可能比较长），并把数据处理为npy文件，同时生成数据集的info文件。

   ```
   python3 CNN_Transformer_Preprocess.py --pre_data_save_path=./pre_data/validation --which_dataset=validation
   ```

   + 参数说明：
     + --pre_data_save_path：预处理数据保存路径；
     + --which_dataset：指定所用的数据集；
         + validation：patrickvonplaten/librispeech_asr_dummy数据集，特别小，只有70多条音频数据；
         + clean：Librispeech clean数据集；
         + other：Librispeech other数据集；
   
   官方提供了模型在Librispeech clean和Librispeech other数据集上的精度，本示例中仅用Librispeech validation测试精度。
   
   运行完后会在--pre_data_save_path指定目录生成数据集预处理好的npy文件



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 导出onnx文件。

      使用`CNN_Transformer_test_export_onnx.py`导出onnx文件。

      运行`CNN_Transformer_test_export_onnx.py`脚本，会自动在线下载pth模型，并把pth模型转换为onnx模型。

      ```
      python3 CNN_Transformer_test_export_onnx.py --model_save_dir=./models   
      ```
      + 参数说明：
         + --model_save_dir：导出的onnx模型文件保存地址；
      
      运行完后会在--model_save_dir指定的目录下生成`wav2vec2-base-960h.onnx`模型文件。


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
         atc --framework=5 --model=./models/wav2vec2-base-960h.onnx --output=./models/wav2vec2-base-960h --input_format=ND --input_shape="input:1,-1" --dynamic_dims="10000;20000;30000;40000;50000;60000;70000;80000;90000;100000;110000;120000;130000;140000;150000;160000;170000;180000;190000;200000;210000;220000;230000;240000;250000;260000;270000;280000;290000;300000;310000;320000;330000;340000;350000;360000;370000;380000;390000;400000;410000;420000;430000;440000;450000;460000;470000;480000;490000;500000;510000;520000;530000;540000;550000;560000" --log=error --soc_version=Ascend${chip_name}  --precision_mode=allow_fp32_to_fp16
         ```

         - 参数说明：

           -   --model：为ONNX模型文件；
           -   --framework：5代表ONNX模型；
           -   --output：输出的OM模型；
           -   --input\_format：输入数据的格式；
           -   --input\_shape：输入数据的shape；
           -   --log：日志级别；
           -   --soc\_version：处理器型号；

          运行成功后在--output指定地址生成wav2vec2-base-960h.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请点击本链接进行安装ais_bench推理工具，以及查看具体使用方法(https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)  

   2. 创建推理输出文件夹

      ```
      mkdir ${out_dir} 
      ```

   3. 执行推理。

        ```
        python3 -m ais_bench --model ./models/wav2vec2-base-960h.om --input ./pre_data/validation/ --auto_set_dymdims_mode 1 --output ${out_dir}  
        ```

        -   参数说明：

             -   --model：om文件路径；
             -   --input：预处理完的数据集文件夹；
             -   --output：推理结果保存地址；
             -   --auto_set_dymdims_mode：自动设置动态维度模式；
      
        `${out_dir}`为推理输出文件夹
      
         推理完成后默认在--output指定目录下生成推理结果。其目录命名格式为xxxx_xx_xx-xx_xx_xx(年_月_日-时_分_秒)，如2022_11_14-16_13_58。

         >**说明：** 
         >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]。

   3. 精度验证。

      调用`CNN_Transformer_Postprocess.py`脚本，可以获得精度统计数据。

      ```
      python3 CNN_Transformer_Postprocess.py \
      --bin_file_path=./om_infer_res_clean/2022_11_14-16_13_58/ \
      --res_save_path=./om_infer_res_clean/transcriptions.txt \
      --which_dataset=validation
      ```

      - 参数说明：
        - --bin_file_path：ais_bench工具推理结果存放路径；
        - --res_save_path：后处理结果存放txt文件;
        - --which_dataset：精度统计所用的数据集，参看CNN_Transformer_Preprocess.py的参数说明；
      
      注：--bin_file_path指定的result_bs16/2022_09_01-18_51_23/路径不是固定，具体路径为ais_bench工具推理命令中，--output指定目录下的生成推理结果所在路径

      精度验证完成后，在--res_save_path指定目录下生成后处理结果存放的txt文件，同时在txt文件的同级目录下生成精度统计文件wer.txt 

   4. 性能验证。

      由于TensorRT无法运行wav2vec2-base-960h.onnx模型，所以性能测试以ais_bench工具得到的om推理性能和pytorch在线推理性能作比较。

      在GPU环境上运行`CNN_Transformer_pth_online_infer.py`脚本，得到pytorch在线推理性能。

        ```
        python3 CNN_Transformer_pth_online_infer.py \
        --pred_res_save_path=./pth_online_infer_res/validation/transcriptions.txt \
        --which_dataset=validation
        ```

      - 参数说明：
         - --pred_res_save_path：pytorch在线推理结果存放路径；
         - --which_dataset：参看CNN_Transformer_Preprocess.py的参数说明;



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据，其中wer为词错率，acc=1-wer

注：模型仅支持bs1,动态分档

| Precision |             |
| --------- | ----------- |
| 标杆精度  | wer=0.0565  |
| 310P3精度 | wer=0.05565 |

| 芯片型号 | Batch Size | 数据集                 | 性能       |
| -------- | ---------- | ---------------------- | ---------- |
| 310P     | 1          | Librispeech validation | 52.1046fps |

