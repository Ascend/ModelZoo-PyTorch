# ESPnet2 for PyTorch
- [概述](#概述)
- [准备训练环境](#准备训练环境)
- [开始训练](#开始训练)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)

# 概述
ESPNet是一套基于E2E的开源工具包，可进行语音识别等任务。从另一个角度来说，ESPNet和HTK、Kaldi是一个性质的东西，都是开源的NLP工具；引用论文作者的话：ESPnet是基于一个基于Attention的编码器-解码器网络，另包含部分CTC组件。

- 参考实现：

  ```
  url=https://github.com/espnet/espnet/tree/v.0.10.5
  commit_id=b053cf10ce22901f9c24b681ee16c1aa2c79a8c2
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/audio/
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  | PyTorch 1.11 | - |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。

  ```
  pip3 install -r requirements.txt
  ```


- 安装ESPnet。

  1. 安装好相应的cann包、pytorch和apex包，并设置好pytorch运行的环境变量；

  2. 基于espnet官方的安装说明进行安装： [Installation — ESPnet 202205 documentation](https://espnet.github.io/espnet/installation.html) 

  安装过程比较复杂，需注意以下几点：

  - 安装依赖的软件包时，当前模型可以只安装cmake/sox/sndfile；

  - 跳过安装kaldi；

  - 安装espnet时，步骤1中的git clone ESPnet代码替换为下载本modelzoo中ESPnet的代码；步骤2跳过；步骤3中设置python环境，若当前已有可用的python环境，可以选择D选项执行；步骤4中进入tools目录后，需要增加installers文件夹的执行权限```chmod +x -R installers/```，然后直接使用make命令进行安装，不需要指定PyTorch版本；
  
  - custom tool installation这一步可以选择不安装。check installation步骤在make时已执行，可跳过；
  
  3. 运行模型前，还需安装：

  - boost: ubuntu上可使用 ```apt install libboost-all-dev```命令安装，centos上使用 ```yum install boost-devel``` 命令安装。
  - kenlm：进入<espnet-root>/tools目录，执行`make kenlm.done`
  
  4. 更新软连接：
  
      ```
      cd <espnet-root>/egs2/aishell/asr1
      rm -f asr.sh db.sh path.sh pyscripts scripts utils steps local/download_and_untar.sh
      ln -s ../../TEMPLATE/asr1/asr.sh asr.sh
      ln -s ../../TEMPLATE/asr1/db.sh db.sh
      ln -s ../../TEMPLATE/asr1/path.sh path.sh
      ln -s ../../TEMPLATE/asr1/pyscripts pyscripts
      ln -s ../../TEMPLATE/asr1/scripts scripts
      ln -s ../../../tools/kaldi/egs/wsj/s5/utils utils
      ln -s ../../../tools/kaldi/egs/wsj/s5/steps steps
      ln -s ../../../../egs/aishell/asr1/local/download_and_untar.sh local/download_and_untar.sh
      ```
      
  5. 增加执行权限：
  
     ```
     chmod +x -R ../../TEMPLATE/asr1
     chmod +x ../../../egs/aishell/asr1/local/download_and_untar.sh
     chmod +x -R local
     chmod +x run.sh
     ```


## 准备数据集

1. 获取数据集。

   本次训练采用**aishell-1**数据集，该数据集包含由 400 位说话人录制的超过 170 小时的语音，数据集目录结构参考如下所示。

   ```
   /downloads
          ├── data_aishell
          ├── data_aishell.tgz
          ├── resource_aishell
          └── resource_aishell.tgz
   ```
   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。
   
   启动训练脚本stage 1 时自行下载并解压数据，下载时间较长，请耐心等待。 如果本地已有aishell数据集，可通过如下软连接命令进行指定。
   
   ```ln -s ${本地aishell数据集文件夹}/ downloads```


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练

     ```
     bash ./test/train_full_1p.sh --stage=起始stage  # 单卡精度
    
     bash ./test/train_performance_1p.sh --stage=起始stage  # 单卡性能
     ```
   
   - 单机8卡训练

     启动8卡训练
   
     ```
     bash ./test/train_full_8p.sh --stage=起始stage  # 8卡精度
    
     bash ./test/train_performance_8p.sh --stage=起始stage  # 8卡性能
     ```
   
   --fp32开启FP32模式

3. 启动训练后，日志输出路径为：<espnet-root>/egs2/aishell/asr1/nohup.out ，该日志中会打印二级日志（各个stage日志）的相对路径。
如：stage 11 的日志路径为：“exp/asr_train_asr_conformer_raw_zh_char_sp/train.log”

模型训练脚本参数说明如下。

```shell
--stage   # 可选参数，默认为1，可选范围为：1~16。后续stage依赖前序stage，首次训练需从stage1开始。 
# stage 1 ~ stage 5 数据集下载与准备
# stage 6 ~ stage 9 语言模型训练
# stage 10 ~ stage 11 ASR模型训练
# stage 12 ~ stage 13 在线推理及精度统计
# stage 14 ~ stage 16 模型打包及上传
```


# 训练结果展示

**表 2**  训练结果展示表

| NAME          | 精度模式 | CER   | FPS    | Epochs | Torch_version |
|--------       |-------------|:-------|--------| :------------ |--------       |
| 1p-竞品       | 混合精度       | -           | 196.86 | 1      | -             |
| 8p-竞品       | 混合精度    | 95.4        | 398.8  | 50     | -             |
| 1p-NPU(非ARM)   | 混合精度       | -           | 101.04 | 1      | 1.8           |
| 8p-NPU(非arm) | 混合精度    | 95.5        | 399.7  | 50     | 1.8           |
| 8p-NPU(ARM)   | 混合精度    | 95.4        | 540  | 50     | 1.8           |
| 8p-NPU(ARM) | FP32 | 95.4 | 670 | 50 | 1.8 |


# 版本说明

## 变更

2023.03.13：更新readme，重新发布。

2022.08.17：首次发布。

## FAQ

无。


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# Conformer-推理指导

- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能&精度](#模型推理性能&精度)

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
  | 固件与驱动                                                      | 22.0.3 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            |  6.0.0 | -                                                                                                     |
  | Python                                                          |  3.7.5 | -                                                                                                     |
  | PyTorch                                                         | 1.13.0 | -                                                                                                     |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 |      \ | \                                                                                                     |


# 快速上手
可参考实现https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/audio/Conformer_for_Pytorch

## 获取源码

1. 获取`Pytorch`源码  
   ```
   git clone https://github.com/espnet/espnet_onnx.git
   cd espnet_onnx
   git reset --hard 18eb341
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
   下载权重放在Conformer_for_Pytorch目录下。权重链接：https://github.com/espnet/espnet/tree/master/egs2/aishell/asr1
   
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
   导出的onnx文件正常在/root/.cache/espnet_onnx/asr_train_asr_qkv/full目录下，在/root/.cache/espnet_onnx/asr_train_asr_qkv目录下则有配置文件config.yaml配置文件以及feats_stats.npz文件
   
   修改导出的`onnx`模型，修改xformer_decoder.onnx文件以及transformer_lm.onnx文件，原因是两模型中存在Gather算子indices为-1的场景，当前CANN还不支持该场景，有精度问题，并且可以优化部分性能。
   
   ```
   python3 modify_onnx_decoder.py /root/.cache/espnet_onnx/asr_train_asr_qkv/full/xformer_decoder.onnx \
   /root/.cache/espnet_onnx/asr_train_asr_qkv/full/xformer_decoder_revise.onnx
   python3 modify_onnx_lm.py /root/.cache/espnet_onnx/asr_train_asr_qkv/full/transformer_lm.onnx \
   /root/.cache/espnet_onnx/asr_train_asr_qkv/full/transformer_lm_revise.onnx
   python3 modify_onnx_ctc.py /root/.cache/espnet_onnx/asr_train_asr_qkv/full/ctc.onnx \
   /root/.cache/espnet_onnx/asr_train_asr_qkv/full/ctc_dynamic.onnx
   python3 modify_onnx_encoder.py /root/.cache/espnet_onnx/asr_train_asr_qkv/full/xformer_encoder.onnx \
   /root/.cache/espnet_onnx/asr_train_asr_qkv/full/xformer_encoder_multibatch.onnx 4

   ```
   
3. 使用`ATC`工具将`ONNX`模型转为`OM`模型  
   
   3.1  执行命令查看芯片名称（得到`atc`命令参数中`soc_version`）
   
   ```
   npu-smi info
   #该设备芯片名为Ascend910A （自行替换）
   回显如下：
   +-------------------|-----------------|------------------------------------------------------+
   | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
   | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
   +===================+=================+======================================================+
   | 0       910A     | OK              | 15.8         42                0    / 0              |
   | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
   +===================+=================+======================================================+
   | 1       910A     | OK              | 15.4         43                0    / 0              |
   | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
   +===================+=================+======================================================+
   ```
   
   3.2  执行ATC命令 
   
   将xformer_encoder.sh，xformer_decoder.sh，transformer_lm.sh，ctc.sh放置到/root/.cache/espnet_onnx/asr_train_asr_qkv/full目录下，运行xformer_encoder.sh导出encoder`OM`模型，默认保存在当前文件夹下，其他模型类似。
   
   ```
    bash xformer_encoder.sh Ascend910A
    bash xformer_decoder.sh Ascend910A
    bash transformer_lm.sh Ascend910A
    bash ctc.sh Ascend910A
   ```

### 2 开始推理验证

1. 修改配置参数

   修改/root/.cache/espnet_onnx/asr_train_asr_qkv/目录下config配置文件参数，给每个模型增加input_size,output_size参数以及修改对应的weight参数中的ctc, decoder, lm，给出样例如下

   | 项          | 子项        |                                                                           路径或值 |
   | :------     | ----------- |                       ------------------------------------------------------------ |
   | encoder     | model_path  |            /root/.cache/espnet_onnx/asr_train_asr_qkv/full/xformer_encoder_rank.om |
   |             | output_size |                                                                           5000000 |
   | decoder     | model_path  | /root/.cache/espnet_onnx/asr_train_asr_qkv/full/xformer_decoder_{os}_{arch}.om |
   |             | output_size |                                                                           5000000 |
   | ctc         | model_path  |             /root/.cache/espnet_onnx/asr_train_asr_qkv/full/ctc_{os}_{arch}.om |
   |             | output_size |                                                                          100000000 |
   | lm          | model_path  |  /root/.cache/espnet_onnx/asr_train_asr_qkv/full/transformer_lm_{os}_{arch}.om |
   |             | output_size |                                                                           5000000 |
   | beam_search | beam_size   |                                                                                  2 |
   | weights     | ctc         |                                                                                0.3 |
   |             | decoder     |                                                                                0.7 |
   |             | lm          |                                                                                0.3 |

   说明：{os}_{arch}为对应系统/架构名，如：{linux}_{aarch64}

2. 执行推理 & 精度验证
   运行`om_val.py`推理OM模型，生成的结果txt文件在当前文件夹下。

   ```
   # 生成的om.txt可以跟标杆对比即可:
   python3 om_val.py --dataset_path ${dataset_path}/wav/test --model_path /root/.cache/espnet_onnx/asr_train_asr_qkv

   # text是标杆文件: 默认打印error值，最终精度取ACC值：100%-error
   python3 compute-wer.py --char=1 --v=1 text om.txt
   ```

3. 性能验证

   打印终端的时间即为数据集上的端到端推理耗时

   模型推理性能&精度:

   调用ACL接口推理计算，性能&精度参考下列数据:
   备注说明：

   1. NPU推理采用多进程推理方案，依赖CPU性能，参考机器：96核CPU(aarch64)/CPU max MHZ: 2600/251G内存/NPU310P3
   
   2. 性能以最终total的端到端性能为准

   | 芯片型号      | 配置                                   | 数据集    |   精度(overall) | 性能(fps)                                  |
   | :-----------: | :------------------------------------: | :-------: | :-------------: | 
   | GPU           | encoder/decoder/ctc/lm(beam_size=20)   | aishell   |          95.27% | 
   | GPU           | encoder/decoder/ctc/lm(beam_size=2)    | aishell   |          95.08% |
   | Ascend910A   | encoder/decoder/ctc/lm(default)        | aishell   |          95.02% | 
