# Wenet模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Wenet模型是一个使用Conformer结构的ASR（语音识别）模型，具有较好的端到端推理精度和推理性能。

- 参考实现：

  ```
  url=https://github.com/Slyne/ctc_decoder.git 
  branch=v2.0.1
  model_name=Wenet
  ```


## 输入输出数据<a name="section540883920406"></a>

- encoder online输入数据

  | 输入数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | chunk_xs | B x 67 x 80 | FLOAT32 | ND |
  | chunk_lens | B | INT32 | ND |
  | offset | B x 1 | INT64 | ND |
  | att_cache | B x 12 x 4 x 64 x 128 | FLOAT32 | ND |
  | cnn_cache | B x 12 x 256 x 7 | FLOAT32 | ND |
  | cache_mask | B x 1 x 64 | FLOAT32 | ND |


- encoder online输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | log_probs | batchsize x Castlog_probs_dim_1 x 10 | FLOAT32  | ND |
  | log_probs_idx | batchsize x Castlog_probs_dim_1 x 10 | INT64 | ND |
  | chunk_out | batchsize x Castlog_probs_dim_1 x 256 | FLOAT32 | ND |
  | chunk_out_lens | batchsize | INT32 | ND |
  | r_offset | batchsize x 1 | INT64 | ND |
  | r_att_cache | batchsize x 12 x dim_2 x dim_3 x dim_4 | FLOAT32 | ND |
  | r_cnn_cache | batchsize x 12 x 256 x dim_5 | FLOAT32 | ND |
  | r_cache_mask | batchsize x 1 x 64 | FLOAT32 | ND |

- encoder offline输入数据

  | 输入数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | speech | batchsize x T x 80 | FLOAT32 | ND |
  | speech_lengths | batchsize | INT32 | ND |


- encoder offline输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | encoder_out | batchsize x T_out x 256 | FLOAT32 | ND |
  | encoder_out_lens | batchsize | INT32 | ND |
  | ctc_log_probs | batchsize x T_OUT x 4233 | FLOAT32 | ND |
  | beam_log_probs | batchsize x T_OUT x 10 | FLOAT32 | ND |
  | beam_log_probs_idx | batchsize x T_OUT x 10 | INT64 | ND |


- decoder输入数据

  | 输入数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | encoder_out | batchsize x T x 256 | FLOAT32 | ND |
  | encoder_out_lens | batchsize | INT32 | ND |
  | hyps_pad_sos_eos | batchsize x 10 x T2 | INT64 | ND |
  | hyps_lens_sos | batchsize x 10 | INT32 | ND |
  | r_hyps_pad_sos_eos | batchsize x 10 x T2 | INT64 | ND |
  | ctc_score | batchsize x 10 | FLOAT32 | ND |


- decoder输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | best_index | batchsize | INT64    | ND           |


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
   
   在工作目录下执行下述命令获取源码并切换到相应路径。

   ```
    git clone https://github.com/wenet-e2e/wenet.git
    cd wenet
    git checkout v2.0.1
    cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

3. 安装om_gener

   ```
   git clone https://gitee.com/peng-ao/om_gener.git
   cd om_gener
   pip3 install .
   ```

4. 安装ctc_dcoder

   ```
   git clone https://github.com/Slyne/ctc_decoder.git
   apt-get update
   apt-get install swig
   apt-get install python3-dev 
   cd ctc_decoder/swig && bash setup.sh
   ```

5. 安装acl_infer

   ```
   git clone https://gitee.com/peng-ao/pyacl.git
   cd pyacl
   pip3 install .
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   ```
   cd wenet/examples/aishell/s0/
   mkdir -p /export/data/asr-data/OpenSLR/33
   bash run.sh --stage -1 --stop_stage -1 # 下载数据集
   ```

2. 处理数据集。

   ```
   bash run.sh --stage 0 --stop_stage 0
   bash run.sh --stage 1 --stop_stage 1
   bash run.sh --stage 2 --stop_stage 2
   bash run.sh --stage 3 --stop_stage 3
   ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      WeNet预训练模型[下载链接](https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.md)，选择aishell数据集对应的Checkpoint Model下载，将压缩文件放到自己克隆wenet仓库的路径下解压。
      ```
      cd ../../../
      tar -zvxf aishell_u2pp_conformer_exp.tar.gz
      mkdir -p exp/aishell_u2pp_conformer_exp
      cp examples/aishell/s0/data/test/data.list aishell_u2pp_conformer_exp
      cp examples/aishell/s0/data/test/text aishell_u2pp_conformer_exp
      cp aishell_u2pp_conformer_exp/global_cmvn exp/aishell_u2pp_conformer_exp
      cp ../export_onnx_npu.py wenet/bin
      cp ../recognize_om.py wenet/bin
      cp ../cosine_similarity.py .
      ```

   2. 导出onnx文件。

      使用export_onnx_npu.py导出onnx文件。

      配置python path
      ```
      export PYTHONPATH=${your_wenet_path}
      ```

      ```
      # 非流式场景
      python3 wenet/bin/export_onnx_npu.py --config aishell_u2pp_conformer_exp/train.yaml --checkpoint aishell_u2pp_conformer_exp/final.pt --output_onnx_dir ./onnx/ --num_decoding_left_chunks 4 --reverse_weight 0.3
      
      # 流式场景
      python3 wenet/bin/export_onnx_npu.py --config aishell_u2pp_conformer_exp/train.yaml --checkpoint aishell_u2pp_conformer_exp/final.pt --output_onnx_dir ./onnx/ --num_decoding_left_chunks 4 --reverse_weight 0.3 --streaming
      ```

      - 参数说明：

         - --config：aishell预训练模型配置文件路径。
         - --checkpoint：aishell预训练模型checkpoint文件路径。
         - --output_onnx_dir：输出onnx的文件夹路径。
         - --num_decoding_left_chunks：self-attention依赖的左侧chunk数。
         - --reverse_weight：从右往左解码权重。
         - --streaming：流式开关。

      获得offline_encoder.onnx、online_encoder.onnx、decoder.onnx文件。

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
         # 流式场景encoder部分
         atc --input_format=ND --framework=5 --model=onnx/online_encoder.onnx --input_shape="chunk_xs:${batch_size},67,80;chunk_lens:${batch_size};offset:${batch_size},1;att_cache:${batch_size},12,4,64,128;cnn_cache:${batch_size},12,256,7;cache_mask:${batch_size},1,64" --output=om/online_encoder_bs${batch_size} --log=error --soc_version=Ascend${chip_name}
         
         # 非流式动态shape场景encoder部分
         atc --input_format=ND --framework=5 --model=onnx/offline_encoder.onnx --input_shape="speech:[1~64,1~1500,80];speech_lengths:[1~64]" --output=om/offline_encoder_dynamic --log=error --soc_version=Ascend${chip_name}
         
         # 非流式分档场景encoder部分
         atc --input_format=ND --framework=5 --model=onnx/offline_encoder.onnx --input_shape="speech:${batch_size},-1,80;speech_lengths:${batch_size}" --dynamic_dims="262;326;390;454;518;582;646;710;774;838;902;966;1028;1284;1478" --output=om/offline_encoder_static_bs${batch_size} --log=error --soc_version=Ascend${chip_name}
         
         # 非流式分档场景decoder部分
         bash static_decoder.sh ${batch_size} Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --input\_shape\：输入数据的shape范围。
           -   --dynamic\_dims：设置ND格式下动态维度的档位。 
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

         运行成功后生成online_encoder_bs${batch_size}.om、offline_encoder_dynamic.om、offline_encoder_static_bs${batch_size}.om、offline_decoder_static_bs${batch_size}.om 模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 非流式分档场景精度和性能验证。

      端到端encoder + decoder

      ```
      python3 wenet/bin/recognize_om.py --config=aishell_u2pp_conformer_exp/train.yaml --test_data=aishell_u2pp_conformer_exp/data.list --dict=aishell_u2pp_conformer_exp/units.txt --mode=attention_rescoring --result_file=static_result_bs${batch_size}.txt --encoder_om=om/offline_encoder_static_bs${batch_size}.om --decoder_om=om/offline_decoder_static_bs${batch_size}.om --batch_size=${batch_size} --device_id=0 --static --test_file=static_test_result_bs${batch_size}.txt
      ```

      - 参数说明：
        - --config：aishell预训练模型配置文件路径。
        - --test_data：测试数据路径。
        - --dict：aishell预训练模型词典路径。
        - --mode：解码模式，可选ctc_greedy_search、ctc_prefix_beam_search和attention_rescoring。
        - --result_file：解码结果文件。
        - --encoder_om：非流式分档场景encoder的om路径。
        - --decoder_om：非流式分档场景decoder的om路径。
        - --batch_size：batch大小。
        - --device_id：卡序号。
        - --static：是否执行分档模式。
        - --test_file：性能结果文件。

      ```
      # 精度验证
      python3 tools/compute-wer.py --char=1 --v=1 aishell_u2pp_conformer_exp/text static_result_bs${batch_size}.txt
      ```

      - 参数说明：
        - --char：是否逐词比对，0为整句比对，1为逐词比对。
        - --v：是否打印对比结果，0为不打印，1为打印。
        - aishell_u2pp_conformer_exp/text：标签文件路径。
        - static_result.txt：比对结果输出文件路径。

   3. 流式纯推理场景精度和性能验证。

      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      # 性能验证
      python3 -m ais_bench --model=online_encoder_bs${batch_size}.om --loop=1000 --batchsize=${batch_size}
      ```

      - 参数说明：
        - --model：om文件路径。
        - --loop：循环次数。
        - --batchsize：batch大小。
      
      ```
      # 精度验证
      python3 cosine_similarity.py --encoder_onnx=onnx/online_encoder.onnx --encoder_om=om/online_encoder_bs${batch_size}.om --batch_size=${batch_size} --device_id=0
      ```
      
      - 参数说明：
        - --encoder_onnx：流式encoder的onnx路径。
        - --encoder_om：流式encoder的om路径。
        - --batch_size：batch大小。
        - --device_id：卡序号。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

性能参考下列数据。 

非流式分档（encoder + decoder）场景

| 芯片型号        | Batch Size | 数据集       | 精度(WER) | 端到端性能（fps） |
|-------------|------------|-----------|---------|----------------|
| Ascend310P3 | 1          | aishell   | 4.67%   | 36.72          |
| Ascend310P3 | 4          | aishell   | 4.67%   | 42.57          |
| Ascend310P3 | 8          | aishell   | 4.67%   | 51.36          |
| Ascend310P3 | 16         | aishell   | 4.67%   | 52.93          |
| Ascend310P3 | 32         | aishell   | 4.67%   | 58.39          |
| Ascend310P3 | 64         | aishell   | 4.67%   | 51.89          |

流式纯推理场景

| 芯片型号        | Batch Size | om与onnx余弦相似度  | 性能             |
|---------------|------------|-------------------|----------------|
| Ascend310P3   | 1          | 0.9999997         | 468.7643 fps   |
| Ascend310P3   | 4          | 0.9999957         | 1126.4175 fps  |
| Ascend310P3   | 8          | 0.9999978         | 1734.4887 fps  |
| Ascend310P3   | 16         | 0.9999974         | 2015.7208 fps  |
| Ascend310P3   | 32         | 0.9999971         | 2177.3159 fps  |
| Ascend310P3   | 64         | 0.9999974         | 2484.5166 fps  |