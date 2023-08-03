# Conformer-推理指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理精度](#模型推理精度)

******

# 概述
Conformer是将CNN用于增强Transformer来做ASR的结构

- 版本说明：
  ```
  url=https://github.com/espnet/espnet_onnx
  commit_id=18eb341
  model_name=Conformer
  ```

# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

  | 配套                                                            |   版本 | 环境准备指导                                                                                          |
  | ------------------------------------------------------------    | ------ | ------------------------------------------------------------                                          |
  | 固件与驱动                                                       | 22.0.3 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            |  6.3.RC2 | -                                                                                                   |
  | Python                                                          |  3.7.5 | -                                                                                                     |
  | PyTorch                                                         | 1.13.0 | -                                                                                                     |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 |      \ | \                                                                                                     |


# 快速上手

## 获取源码

1. 获取`Pytorch`源码  
   ```
   git clone https://github.com/espnet/espnet_onnx.git
   cd espnet_onnx
   git reset --hard 18eb341
   cd ..
   ```
   
2. 安装依赖  
   ```
   pip3 install -r requirements.txt
   ```
   
3. 安装ais-bench/auto-optimizer

   参考[ais-bench](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)/[auto-optimizer](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer/auto_optimizer)安装。
   

4. 获取`OM`推理代码

   目录结构如下：

   ```
   ├──Conformer_for_Pytorch
      ├── pth2onnx.py
      ├── modify_onnx_lm.py
      ├── modify_onnx_decoder.py
      ├── graph_fusion.py
      ├── export_acc.patch
      ├── espnet_onnx
      ├── ...
   ```


## 准备数据集
- 该模型使用AISHELL数据集进行精度评估，下载[aishell数据集](https://zhuanlan.zhihu.com/p/535316385)


## 模型推理

### 1 模型转换

将模型权重文件`.pth`转换为`.onnx`文件，再使用`ATC`工具将`.onnx`文件转为离线推理模型`.om`文件。

1. 获取权重文件  
   下载权重放在Conformer_for_Pytorch目录下。权重链接：https://zenodo.org/record/4105763#.X40xe2j7QUE
   
   指定参数为：Conformer + specaug + speed perturbation: feats=raw, n_fft=512, hop_length=128
   
   点击With Transformer LM中的Model link链接下载asr_train_asr_conformer3_raw_char_batch_bins4000000_accum_grad4_sp_valid.acc.ave.zip文件，将该文件和pth2onnx.py文件置于同一目录下


2. 导出`ONNX`模型    
   
   ```
   cd espnet_onnx
   patch -p1 < ../export_acc.patch
   cp ../multi_batch_beam_search.py espnet_onnx/asr/beam_search
   cp ../asr_npu_adapter.py espnet_onnx/asr
   cp ../npu_model_adapter.py espnet_onnx/asr
   pip3 install .  #安装espnet_onnx
   cd ..
   ```
   配置环境变量  
   
   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
   
   > **说明：**  
   > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
   
   运行`pth2onnx.py`导出`ONNX`模型。  

   ```
   python3 pth2onnx.py
   ```
   导出的onnx文件正常在${HOME}/.cache/espnet_onnx/asr_train_asr_qkv/full目录下，在${HOME}/.cache/espnet_onnx/asr_train_asr_qkv目录下则有配置文件config.yaml配置文件以及feats_stats.npz文件
   
   修改导出的`onnx`模型，修改xformer_decoder.onnx、ctc.onnx、xformer_encoder.onnx文件，使模型支持多batch推理。
   
   ```
   python3 modify_onnx_decoder.py ${HOME}/.cache/espnet_onnx/asr_train_asr_qkv/full/xformer_decoder.onnx \
   ${HOME}/.cache/espnet_onnx/asr_train_asr_qkv/full/xformer_decoder_revise.onnx
   python3 modify_onnx_ctc.py ${HOME}/.cache/espnet_onnx/asr_train_asr_qkv/full/ctc.onnx \
   ${HOME}/.cache/espnet_onnx/asr_train_asr_qkv/full/ctc_dynamic.onnx
   python3 modify_onnx_encoder.py ${HOME}/.cache/espnet_onnx/asr_train_asr_qkv/full/xformer_encoder.onnx \
   ${HOME}/.cache/espnet_onnx/asr_train_asr_qkv/full/xformer_encoder_multibatch.onnx 24
   ```
   
3. 使用`ATC`工具将`ONNX`模型转为`OM`模型  
   
   3.1  执行命令查看芯片名称（得到`atc`命令参数中`soc_version`）
   
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
   
   3.2  执行ATC命令 
   
   将xformer_encoder.sh，xformer_decoder.sh，transformer_lm.sh，ctc.sh放置到${HOME}/.cache/espnet_onnx/asr_train_asr_qkv/full目录下，运行xformer_encoder.sh导出encoder`OM`模型，默认保存在当前文件夹下，其他模型类似。
   
   ```
    bash xformer_encoder.sh Ascend310P3 24
    bash xformer_decoder.sh Ascend310P3
    bash transformer_lm.sh Ascend310P3
    bash ctc.sh Ascend310P3
   ```

### 2 开始推理验证

1. 修改配置参数

   修改${HOME}/.cache/espnet_onnx/asr_train_asr_qkv/目录下config配置文件参数，给每个模型增加input_size,output_size参数以及修改对应的weight参数中的ctc, decoder, lm，给出样例如下

   | 项          | 子项        |                                                                           路径或值 |
   | :------     | ----------- |                       ------------------------------------------------------------ |
   | encoder     | model_path  |            ${HOME}/.cache/espnet_onnx/asr_train_asr_qkv/full/xformer_encoder_rank.om |
   |             | output_size |                                                                           5000000 |
   | decoder     | model_path  | ${HOME}/.cache/espnet_onnx/asr_train_asr_qkv/full/xformer_decoder_{os}_{arch}.om |
   |             | output_size |                                                                           5000000 |
   | ctc         | model_path  |             ${HOME}/.cache/espnet_onnx/asr_train_asr_qkv/full/ctc_{os}_{arch}.om |
   |             | output_size |                                                                          100000000 |
   | lm          | model_path  |  ${HOME}/.cache/espnet_onnx/asr_train_asr_qkv/full/transformer_lm_{os}_{arch}.om |
   |             | output_size |                                                                           5000000 |
   | beam_search | beam_size   |                                                                                  2 |
   | weights     | ctc         |                                                                                0.3 |
   |             | decoder     |                                                                                0.7 |
   |             | lm          |                                                                                0.3 |

   说明：{os}_{arch}为对应系统/架构名，如：{linux}_{aarch64}

2. 执行推理 & 精度验证
   运行`om_val.py`推理OM模型，生成的结果txt文件在当前文件夹下。

   ```shell
   # 生成的om.txt可以跟标杆对比即可:
   python3 om_val.py --dataset_path ${dataset_path}/wav/test --model_path ${HOME}/.cache/espnet_onnx/asr_train_asr_qkv --batch_encoder 24 --batch_decoder 24

   # text是标杆文件: 默认打印error值，最终精度取ACC值：100%-error
   python3 compute-wer.py --char=1 --v=1 text om.txt
   ```

# 模型推理精度

   精度参考下列数据:


   | 芯片型号      | 配置                                   | 数据集    |   精度(overall) |
   | :-----------: | :------------------------------------: | :-------: | :-------------: |
   | Ascend310P3   | encoder/decoder/ctc/lm(beam_size=2)    | aishell   |          95.00% |
