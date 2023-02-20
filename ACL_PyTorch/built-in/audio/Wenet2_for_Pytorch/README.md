# Wenet模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)
- [获取源码](#section183221994400)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)
  
- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)






# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Wenet模型是一个使用Conformer结构的ASR（语音识别）模型，具有较好的端到端推理精度和推理性能。

- 参考实现：

  ```
  url=https://github.com/Slyne/ctc_decoder.git 
  branch=v2.0.1
  model_name=Wenet
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




# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.10.1  | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>
安装依赖。

   ```
   pip3 install -r requirements.txt
   ```
om_gener安装

   ```
   git clone https://gitee.com/peng-ao/om_gener.git
   cd om_gener
   pip3 install .
   ```

ctc_dcoder安装

   ```
   git clone https://github.com/Slyne/ctc_decoder.git
   apt-get update
   apt-get install swig
   apt-get install python3-dev 
   cd ctc_decoder/swig && bash setup.sh
   ```

acl_infer安装

   ```
   git clone https://gitee.com/peng-ao/pyacl.git
   cd pyacl
   pip3 install .
   ```



## 获取源码<a name="section183221994400"></a>

在工作目录下执行下述命令获取源码并切换到相应路径。

   ```
    git clone https://github.com/wenet-e2e/wenet.git
    cd wenet
    git checkout v2.0.1
    wenet_path=$(pwd)
   ```

路径说明：

${wenet_path}表示wenet开源模型代码的路径

${code_path}表示modelzoo中Wenet_for_Pytorch工程代码的路径，例如code_path=/home/ModelZoo-PyTorch/ACL_PyTorch/built-in/audio/Wenet_for_Pytorch

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

    cd ${wenet_path}/examples/aishell/s0/

    mkdir -p /export/data/asr-data/OpenSLR/33

    bash run.sh --stage -1 --stop_stage -1 # 下载数据集
    bash run.sh --stage 0 --stop_stage 0 # 处理数据集
    bash run.sh --stage 1 --stop_stage 1 # 处理数据集
    bash run.sh --stage 2 --stop_stage 2 # 处理数据集
    bash run.sh --stage 3 --stop_stage 3 # 处理数据集

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   本模型基于开源框架PyTorch训练的Wenet进行模型转换。使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   下载权重链接：https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.md

   选择aishell数据集对应的Checkpoint Model下载即可

   下载压缩文件，将文件解压，将文件夹内的文件放置到wenet/examples/aishell/s0/exp/20210601_u2++_conformer_exp文件夹下，若没有该文件夹，则创建该文件夹

    1. 放置权重等文件。

        ```
        tar -zvxf 20210601_u2++_conformer_exp.tar.gz
        mkdir -p ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp
        mkdir -p ${wenet_path}/exp/20210601_u2++_conformer_exp
        mkdir -p ${wenet_path}/examples/aishell/s0/onnx/
        cp -r 20210601_u2pp_conformer_exp/20210601_u2++_conformer_exp/* ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp
        cp -r examples/aishell/s0/data/test/data.list ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/
        ```
   ```
        
   ```
        cp -r examples/aishell/s0/data/test/text ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/
        cp -r ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/global_cmvn ${wenet_path}/exp/20210601_u2++_conformer_exp
        cd  ${code_path}
        cp -r export_onnx_npu.py ${wenet_path}/wenet/bin/
        cp -r recognize_om.py ${wenet_path}/wenet/bin/
        cp -r cosine_similarity.py ${wenet_path}/examples/aishell/s0/ 
        cp -r adaptdecoder.py ${wenet_path}/examples/aishell/s0/
        cp -r *.sh ${wenet_path}/examples/aishell/s0/
        ```

   2. 导出onnx文件。

      1. 运行以下命令导出对应的onnx。
         ```
         cd ${wenet_path}/examples/aishell/s0
         mkdir onnx
         #非流式
         python3 wenet/bin/export_onnx_npu.py --config ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/train.yaml --checkpoint ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/final.pt --output_onnx_dir ./onnx/ --num_decoding_left_chunks 4 --reverse_weight 0.3 
         #流式
         python3 wenet/bin/export_onnx_npu.py --config ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/train.yaml --checkpoint ${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/final.pt --output_onnx_dir ./onnx/ --num_decoding_left_chunks 4 --reverse_weight 0.3 --streaming
         ```

2. 使用ATC工具将ONNX模型转OM模型。

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
        cd ${code_path}
        cp ${wenet_path}/examples/aishell/s0/onnx/* ${code_path}/
        bash online_encoder.sh Ascend${chip_name} 
        bash offline_encoder.sh Ascend${chip_name} # Ascend310P3
        bash static_encoder.sh Ascend${chip_name}
        #若需要decoder部分,对于分档场景
        python3 adaptdecoder.py
        bash static_decoder.sh Ascend${chip_name}
        ```


3. 开始推理验证。

   使用ais-infer工具进行推理验证流式模型性能。

   ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   run.sh脚本封装了ctc_greedy_search场景的动态shape和动态分档的推理，并将性能和精度分别保存在了offline_test_result.txt online_test_result.txt以及static_test_result.txt中，

   运行bash run.sh Ascend310P3即可

   首先设置日志等级

   ​        1. 设置日志等级 export ASCEND_GLOBAL_LOG_LEVEL=3

   ​        2. 拷贝om模型到${wenet_path}/examples/aishell/s0

   ```
   cp ${code_path}/online_encoder.om ${wenet_path}/examples/aishell/s0/
   cp ${code_path}/offline_encoder.om ${wenet_path}/examples/aishell/s0/
   ```

   动态shape场景：

- 非流式场景下推理模型


```
python3 wenet/bin/recognize_om.py --config=${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/train.yaml --test_data=${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/data.list --dict=units.txt --mode=ctc_greedy_search --result_file=offline_res_result.txt --encoder_om=encoder_offline.om --decoder_om=xx.om --batch_size=1 --device_id=0 --test_file=offline_test_result.txt
```

- 计算并查看overall精度

  ```
  python3 tools/compute-wer.py --char=1 --v=1 text offline_res_result.txt > offline_wer
  cat offline_wer | grep "Overall"
  ```

- 查看非流式性能

  性能和精度保存在offline_dynamic_results.txt

流式场景

- 获取流式场景下性能

  可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

  ```
     python3 -m ais_bench --model=${om_model_path} --loop=1000 --batchsize=${batch_size}
  ```

  - 参数说明：
    - --model：om模型
    - --batchsize：模型batchsize
    - --loop: 循环次数

- 查看精度(余弦相似度)

  ```
  python3 cosine_similarity.py
  ```

分档场景：

- 推理模型:


```
python3 wenet/bin/recognize_om.py --config=${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/train.yaml --test_data=${wenet_path}/examples/aishell/s0/exp/20210601_u2++_conformer_exp/data.list --dict=units.txt --mode=ctc_greedy_search --result_file=static_res_result.txt --encoder_om=encoder_static.om --decoder_om=decoder_static.om --batch_size=32--device_id=0 --static --test_file=static_test_result.txt
```

- 查看性能和精度
  在static_test_result.txt文件中

# 模型推理性能<a name="ZH-CN_TOPIC_0000001172201573"></a>

性能参考下列数据，因为GPU上decoder部分存在性能bug(比onnxruntime还慢)，我们仅提供encoder部分（对应ctc_greedy_search和ctc_prefix_beam_search）的性能，在数据测试集上的打点性能(最优batch)。

非流式

| 模型              | 310P性能   | T4性能     | 310P/T4 |
|-----------------|----------|----------|---------|
| Wenet动态shape bs=32） | 40s      | 42s    | 1.05 |
| Wenet分档 (bs=32) | 37s | 42s | 1.13 |

流式(纯推理)

| 模型           | 310P性能 | T4性能  | 310P/T4 |
| -------------- | -------- | ------- | ------- |
| Wenet（bs=64） | 2439fps  | 1714fps | 1.42    |

