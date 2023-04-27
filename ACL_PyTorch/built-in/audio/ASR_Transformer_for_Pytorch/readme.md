# ASR-Transformer模型推理指导

- [推理环境](#推理环境)
- [快速上手](#快速上手)
    - [获取源码](#获取源码)
    - [准备数据集](#准备数据集)
    - [模型转换](#模型转换)
    - [推理验证](#推理验证)
- [性能&精度](#性能精度)

----
+ 参考实现：  
    https://huggingface.co/speechbrain/asr-transformer-aishell


    


----
# 推理环境

- 该模型推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | 固件与驱动 | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 6.3.RC1 | -          |
    | Python    | 3.7.5   | -          |
    
    说明：请根据推理卡型号与 CANN 版本选择相匹配的固件与驱动版本。


----
# 快速上手

## 安装


    ```
- 获取源码
    ```bash
    git clone https://github.com/speechbrain/speechbrain/
    cd speechbrain
    git reset --hard e353e95ecc4580b4a79248d00bbb53850017cbf0
    git apply ../transformer.patch
  
    ```

- 安装推理过程所需的依赖
    ```bash
    pip3 install speechbrain
    pip install -r requirements.txt
    pip install -e .
    cd ..
    ```

## 准备数据集

1. 获取原始数据集  
    本模型推理项目使用 aishell-1数据集及rirs_noises.zip验证模型精度，请自行下载，解压后，目录结构如下：
    ```
    data_aishell
    ├── transcript
    │   ├── aishell_transcript_v0.8.txt
    ├── wav
    │   ├── train
    │   ├── dev
    │   ├── test
    │   │   ├── S0768
    │   │   ├── S0769
    │   │   ├── S0770
    │   │   ├── S0771
    │   │   ├── S0772
    │   │   │   ├── BAC009S0765W0121.wav
    ....
    │   │   │   ├── BAC009S0765W0156.wav
    RIRS_NOISES
    ├── pointsource_noises
    ├── simulated_rirs
    ├── real_rirs_isotropic_noises
    ```




## 模型转换

1. PyTroch 模型转 ONNX 模型  

    请与开源仓（[https://huggingface.co/speechbrain/asr-transformer-aishell]）下载googledrive中的权重文件AISHELL-1-20230413T041318Z-001.zip
    解压zip包，并将解压目录下的result文件夹移动到代码根目录下，删除results下存在的csv文件
    ```
    unzip AISHELL-1-20230413T041318Z-001.zip
    mv AISHELL-1/ASR/Transformer/AISHELL-Transformer/ASR/results/ ./
    rm results/*.csv 
    ```

    修改results/transformer/8886/hyperparams.yaml文件中的tokenizer_file及data_folder为相对应的路径
    将源码仓代码路径添加到PYTHONPATH中，命令如下：
    ```
    export PYTHONPATH=/home/speechbrain/:{PYTHONPATH}
    ```

    然后执行执行以下命令生成 ONNX 模型：
    ```
    python3 infer_and_export.py results/transformer/8886/hyperparams.yaml --device cpu --mode "export" 
    ```
    参数说明：
     + --device : 选择使用的设备。
     + --mode : 生成ONNX模

2. ONNX 模型转 OM 模型  

    step1: 查看NPU芯片名称 \${chip_name}
    ```bash
    npu-smi info
    ```
    例如该设备芯片名为 310P3，回显如下：
    ```
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

    step2: ONNX 模型转 OM 模型
    ```bash
    # 配置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    chip_name=310P3  # 根据 step1 的结果设值
 
    # 执行 ATC 进行模型转换
    atc --model=./encoder.onnx \
        --framework=5 \
        --output=encoder \
        --input_format=ND \
        --input_shape_range="src:[-1,-1,20,256];wav_lens:[-1]" \
        --log=error \
        --soc_version=Ascend${chip_name}

    atc --model=./decoder.onnx \
        --framework=5 \
        --output=decoder \
        --input_format=ND \
        --input_shape_range="token:[-1,-1,256];encoder_out:[-1,-1,256];decoder_mask:[-1,-1]" \
        --log=error \
        --soc_version=Ascend${chip_name}
    ```

   参数说明：
    + --framework: 5代表ONNX模型
    + --model: ONNX模型路径
    + --input_shape: 模型输入数据的shape
    + --input_format: 输入数据的排布格式
    + --output: OM模型路径，无需加后缀
    + --log：日志级别
    + --soc_version: 处理器型号



## 推理验证

1. 对数据集推理  
    python3 infer_and_export.py results/transformer/8886/hyperparams.yaml --device cpu --mode "infer" \
--npu_rank 0 --encoder_file encoder.om --decoder_file decoder.om 
    ```
    参数说明：
    + --device :固定传入参数“cpu”
    + --mode :运行模式，可选infer或export
    + --npu_rank :推理使用的npu设备
    + --encoder_file:encoder文件路径
    + --decoder_file:decoder文件路径


