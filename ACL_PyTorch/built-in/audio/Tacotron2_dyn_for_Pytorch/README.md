# Tacotron2_dyn推理指导

- [Tacotron2\_dyn推理指导](#tacotron2_dyn推理指导)
- [概述](#概述)
    - [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
    - [1 模型转换](#1-模型转换)
    - [2 开始推理验证](#2-开始推理验证)
- [模型推理性能\&精度](#模型推理性能精度)

******


# 概述
Tacotron2是由Google Brain在2017年提出来的一个End-to-End语音合成框架。模型从下到上可以看作由两部分组成：
1. 声谱预测网络：一个Encoder-Attention-Decoder网络，输入字符序列，用于预测梅尔频谱的帧序列。
2. 声码器（vocoder）：一个WaveNet的修订版，输入预测的梅尔频谱帧序列，用于生成时域波形。

- 版本说明：
  ```
  url=https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2
  commit_id=7ce175430ff9af25b040ffe2bceb5dfc9d2e39ad
  model_name=Tacotron2
  ```

### 输入输出数据

- 输入数据

  | 输入数据 | 数据类型 |        大小         | 数据排布格式 |
  | :------: | :------: | :-----------------: | :----------: |
  |   seq    |  INT64   | batchsize x seq_len |      ND      |
  | seq_lens |  INT32   |      batchsize      |      ND      |

- 输出数据

  | 输出数据 | 数据类型 |        大小         | 数据排布格式 |
  | :------: | :------: | :-----------------: | :----------: |
  |   wavs   | FLOAT32  | batchsize x wav_len |      ND      |


# 推理环境准备
- 该模型需要以下插件与驱动  
  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.11.0  | [AscendPytorch安装](https://gitee.com/ascend/pytorch/tree/v1.11.0/) |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


# 快速上手

## 获取源码

1. 获取`Pytorch`源码  
   ```
   git clone https://github.com/NVIDIA/DeepLearningExamples.git
   cd DeepLearningExamples
   git reset --hard 7ce175430ff9af25b040ffe2bceb5dfc9d2e39ad
   cd PyTorch/SpeechSynthesis/Tacotron2
   mkdir -p output/audio  # 新建output文件夹，作为模型结果的默认保存路径
   mkdir checkpoints
   ```
   
2. 安装依赖  
   ```
   pip3 install -r requirements.txt
   ```
   
   由于 onnxsim 存在的一个 [bug](https://github.com/daquexian/onnx-simplifier/issues/214#issuecomment-1315465297)，CPU 架构为 aarch64 的服务器上无法通过 `pip3 install onnxsim==0.4.8` 直接安装 onnxsim，需要从源码编译安装。

   ```bash
   pip3 install --upgrade pip setuptools wheel
   git clone https://github.com/daquexian/onnx-simplifier.git -b v0.4.8
   cd onnx-simplifier
   sed -i 's/git@github.com:/https:\/\/github.com\//g' .gitmodules
   git submodule update --init --recursive -- third_party/onnx-optimizer
   git submodule update --init -- third_party/onnxruntime third_party/pybind11
   python3 setup.py bdist_wheel
   find . -name *.whl -exec pip3 install {} \;
   ```
   
3. 获取`OM`推理代码  
   将推理部署代码放到`Pytorch`源码相应目录下。
   ```
   Tacotron2_dyn_for_PyTorch
    ├── cvt_tacotron2onnx.py  放到Tacotron2/tensorrt下
    ├── cvt_waveglow2onnx.py  放到Tacotron2/tensorrt下
    ├── atc.sh                放到Tacotron2下
    ├── om_val.py             放到Tacotron2下
    ├── val_pyacl.sh          放到Tacotron2下
    └── val_pyacl_cache.sh    放到Tacotron2下
   ```


## 准备数据集
- 该模型使用`LJSpeech`数据集进行精度评估，`Pytorch`源码仓下已包含验证数据（500条文本数据），文件结构如下：
   ```
   filelists
   └── ljs_audio_text_test_filelist.txt
   ```


## 模型推理
### 1 模型转换  
将模型权重文件`.pth`转换为`.onnx`文件，再使用`ATC`工具将`.onnx`文件转为离线推理模型`.om`文件。

1. 获取权重文件  
   `nvidia_tacotron2pyt_fp32_20190427`：[下载地址](https://ngc.nvidia.com/catalog/models/nvidia:tacotron2_pyt_ckpt_fp32)  
   `nvidia_waveglowpyt_fp32_20190427`：[下载地址](https://ngc.nvidia.com/catalog/models/nvidia:waveglow_ckpt_fp32)  
   将下载的模型文件`nvidia_tacotron2pyt_fp32_20190427`和`nvidia_waveglowpyt_fp32_20190427`放在新建的`checkpoints`文件夹下。

2. 导出`ONNX`模型  
   运行`pth2onnx.py`导出`ONNX`模型，结果默认保存在`output/onnx`文件夹下。  
   ```
   python3 tensorrt/cvt_tacotron2onnx.py --tacotron2 ./checkpoints/nvidia_tacotron2pyt_fp32_20190427 -o output/onnx/ -bs 1
   python3 tensorrt/cvt_waveglow2onnx.py --waveglow ./checkpoints/nvidia_waveglowpyt_fp32_20190427 -o output/onnx/ --config-file config.json
   ```

3. 使用`ATC`工具将`ONNX`模型转为`OM`模型  
   3.1 配置环境变量  
   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
   > **说明：**  
   > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

   3.2 执行命令查看芯片名称（得到`atc`命令参数中`soc_version`）
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

   3.3 执行ATC命令  
   运行`atc.sh`导出`OM`模型，默认保存在`output/om`文件夹下。
   ```
   bash atc.sh --soc Ascend310P3 --bs 1
   ```
      - `atc`命令参数说明（参数见`atc.sh`）：
        -   `--model`：ONNX模型文件
        -   `--framework`：5代表ONNX模型
        -   `--output`：输出的OM模型
        -   `--input_format`：输入数据的格式
        -   `--input_shape`：输入数据的shape
        -   `--log`：日志级别
        -   `--soc_version`：处理器型号
        -   `--input_shape_range`：指定模型输入数据的shape范围

    
### 2 开始推理验证

1. 安装`ais_bench`推理工具  
   请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

2. 使能缓存功能（可选）  
   Tacotron2模型的decoder部分有自回归机制，前一步的输出会作为后一步的输入使用。如果使能缓存功能，将中间数据保存到device，可以减少循环推理中间过程中H2D和D2H的数据搬运次数，从而提升端到端的推理性能。  
   这里的缓存功能基于pyACL实现，可参考[pyACL demo](https://gitee.com/peng-ao/pyacl?_from=gitee_search)。  

   2.1 使能CANN的ACL组件  
   参考[CANN V100R020C10 应用软件开发指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100164876/7af18685)，修改环境变量。  

   2.2 下载`pyACL demo`  
   参考[pyACL demo](https://gitee.com/peng-ao/pyacl?_from=gitee_search)，根据readme文档安装demo库。  

3. 执行推理  
   不开启缓存功能时，推荐使用`om_val.py`；开启缓存功能时，推荐使用`val_pyacl_cache.py`。脚本的具体说明请见后。  
   运行`om_val.py`或`val_pyacl_cache.py`推理OM模型，合成语音默认保存在`output/audio`文件夹下。
   ```
   # 推理tacotron2 om。如果选择使能缓存，请将'om_val.py'改为'val_pyacl_cache.py'
   python3 om_val.py -i filelists/ljs_audio_text_test_filelist.txt -bs 1 --device_id 0
    
   # 推理waveglow生成wav文件。如果选择使能缓存，请将'om_val.py'改为'val_pyacl_cache.py'
   python3 om_val.py -i filelists/ljs_audio_text_test_filelist.txt -o output/audio -bs 1 -device_id 0 --gen_wav
   ```
   其中，`bs`为模型`batch_size`，`device_id`设置推理用第几号卡。

4. 性能验证  
   可使用`ais_bench`推理工具的纯推理模式验证不同`batch_size`的`OM`模型的性能。Tacotron2包括多个子模型，各子模型测试性能的参考命令如下：
   ```
   python3 -m ais_bench --model output/om/encoder_dyn_${os}_${arch}.om --loop 20 --batchsize ${bs} --dymShape "sequences:${bs},${seq_len};sequence_lengths:${bs}" --outputSize "3000000,3000000,3000000"
   python3 -m ais_bench --model output/om/decoder_iter_dyn_${os}_${arch}.om --loop 20 --batchsize ${bs} --dymShape "decoder_input:${bs},80;attention_hidden:${bs},1024;attention_cell:${bs},1024;decoder_hidden:${bs},1024;decoder_cell:${bs},1024;attention_weights:${bs},${seq_len};attention_weights_cum:${bs},${seq_len};attention_context:${bs},512;memory:${bs},${seq_len},512;processed_memory:${bs},${seq_len},128;mask:${bs},${seq_len}" --outputSize "20000,20000,20000,20000,20000,20000,20000,20000,20000"
   python3 -m ais_bench --model output/om/postnet_dyn_${os}_${arch}.om --loop 20 --batchsize ${bs} --dymShape "mel_outputs:${bs},80,250" --outputSize "640000"
   ```
   其中，`bs`为模型`batch_size`，`seq_len`为输入音频的长度，`os` 为操作系统（比如 linux），`arch` 为 CPU 架构（比如 x86_64、aarch64）。

5. pyACL demo实现缓存的代码说明  
   我们在pyacl库上，基于ACL实现缓存功能，并开放了相关接口，暂未同步到ais_bench库。  
   如需基于ACL实现缓存功能，需要熟悉ACL提供的python接口，请参见昇腾社区的[CANN文档](https://www.hiascend.com/document/detail/zh/canncommercial/601/inferapplicationdev/aclpythondevg/aclpythondevg_01_0002.html)中 `pyACL API参考`章节。
   这里对[pyACL demo](https://gitee.com/peng-ao/pyacl?_from=gitee_search)中实现缓存功能的方式进行说明。

   5.1 脚本说明  
   - `om_val.py`是整改后调用ais_bench的脚本,推荐使用，不带有缓存接口。  
   - `val_pyacl_cache.py`是调用pyacl库的脚本，开启了缓存功能。  
   - `val_pyacl.py`是调用pyacl库的脚本，未开启缓存功能。其与`om_val.py`功能一致，两者差异点在调用的接口为ais_bench/pyacl。其与`val_pyacl_cache.py`调用接口一致，两者差异点为缓存功能的使能/不使能，为说明缓存接口的使用一并附上。  

   5.2 Demo对外开放接口  
      Demo中与缓存功能相关的接口参数如下，调用方式可参考从val_pyacl.py到val_pyacl_cache.py的改动。  
   - `AclNet`类初始化接口参数说明：
      - `out_to_in`：表示输出输入对应关系，索引的输出数据迭代后缓存在device侧，下次迭代按对应关系传给输入
      - `pin_input`：表示每次迭代都不变的输入索引，索引的数据传入后缓存在device，每次迭代时直接读取，若不存在，可不写
      - `out_idx`：为每次迭代后需要传到device的输出索引，后续一般用于跳出循环的判断条件，若不存在判断条件，可不写  
      - out_to_in和pin_input中指定的中间数据，在推理过程中缓存在Device侧，不会搬运到Host侧
    
   - `AclNet`实例对象的调用接口参数说明：
      - `first_step`：为True时，会传入模型所有输入数据，为False时，只传入除pin_input索引的以外的输入数据
      - `last_step`：为True时，会输出模型所有输出数据，为False时，只传出out_idx索引的输出数据

   5.3 Device侧资源申请  
   - Device侧内存申请  
      模型初始化时，脚本在`_gen_data_buffer`函数中，使用pyACL提供的`acl.rt.malloc`接口申请Device内存。内存地址保存在self.input_data和self.output_data，以供创建`aclDataBuffer`类型的数据。  
   - aclmdlDataset类型的数据创建  
      因为调用pyACL提供的`acl.mdl.execute`接口进行推理时，输入和输出需要为`aclmdlDataset`类型的数据，所以提前从申请的Device内存中创建。  
      具体实现是，调用模型进行推理前，脚本在`_gen_dataset`函数中，使用pyACL提供的`acl.mdl.create_dataset`接口，从self.input_data和self.output_data中保存的Device内存地址创建aclmdlDataset类型的数据，保存在self.load_input_dataset和self.load_input_dataset中，以供后续数据传输和模型推理时使用。  

   5.4 数据搬运  
   调用模型进行推理前后，脚本在`_data_interaction`函数中进行判断，以决定数据在Host和Device之间搬运的方式。  
   - 传入输入数据时  
      - 在`out_to_in`索引中的数据直接在Device侧赋值，即使用缓存；  
      - 在`pin_input`中的数据保持上次迭代的不变，即使用缓存；  
      - 其他数据才涉及Host2Device搬运，调用`acl.rt.memcpy`接口，从Host搬运到self.input_data里保存的Device内存地址。  
   - 传出输出数据时  
      - 只有`out_idx`索引的数据才进行Device2Host搬运，调用`acl.rt.memcpy`接口，从self.input_data中保存的Device内存地址搬运到Host；  
      - 其他数据不搬运。  

   5.5 模型推理  
   脚本在`forward`函数中，调用pyACL提供的`acl.mdl.execute`接口进行推理。输入和输出使用之前创建的`aclmdlDataset`类型的数据。  

# 模型推理性能&精度

调用ACL接口推理计算，性能&精度参考下列数据。

|  芯片型号   | Batch Size |  缓存  |  数据集  |       精度       |    性能     |
| :---------: | :--------: | :------: | :------: | :--------------: | :---------: |
| Ascend310P3 |     1      | Disabled | LJSpeech | 人工判断语音质量 | 441.35 wavs/s  |
| Ascend310P3 |     1      | Enabled | LJSpeech | 人工判断语音质量 | 889.92 wavs/s  |
| Ascend310P3 |     4      | Disabled | LJSpeech | 人工判断语音质量 | 1415.13 wavs/s |
| Ascend310P3 |     4      | Enabled | LJSpeech | 人工判断语音质量 | 3421.21 wavs/s |
- 测试环境：数据在单台x86 CPU服务器下测得，CPU型号为 Intel Xeon Gold 6140 @ 2.30GHz.
- 说明：由于模型推理为多个子模型串联，仅测量单个子模型性能没有意义，故性能采用端到端推理LJSpeech验证集中500条文本数据测得，也就是用 om_val.py 跑出来的性能。