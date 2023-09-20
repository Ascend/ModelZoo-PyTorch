# Unet模型量化指导

## 环境配置
```bash
# 指定量化使用的device
export DEVICE_ID=0

source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

> **说明：** 
>该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

## 量化操作

量化时可使用虚拟数据或者真实数据校准。使用真实数据的量化精度更高，但需进行一次推理得到真实数据。

### 虚拟数据校准

运行quant_unet.py脚本进行量化。

```bash
python3 quant_unet.py \
    --model ${model_base} \
    --model_dir ./models_bs${bs} \
    --prompt_file ./prompts.txt \
    --save_path unet_quant \
    --data_free
```
参数说明：
- --model：模型名称或本地模型目录的路径。
- --model_dir：存放导出模型的目录。
- --prompt_file：输入文本文件，按行分割。
- --save_path：量化模型的储存目录。
- --data_free：使用虚拟数据。

执行成功后生成`models_bs${bs}/unet_quant`文件夹，包含unet.onnx模型及权重。
        
### 真实数据校准
1. 使用ATC工具将ONNX模型转OM模型。

    1. 执行命令查看芯片名称（$\{chip\_name\}）。

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

    2. 执行ATC命令。

        ```bash
        # clip
        atc --framework=5 \
            --model=./models_bs${bs}/clip/clip.onnx \
            --output=./models_bs${bs}/clip/clip \
            --input_format=ND \
            --log=error \
            --soc_version=Ascend${chip_name}
        
        # unet
        cd ./models_bs${bs}/unet/

        atc --framework=5 \
            --model=./unet.onnx \
            --output=./unet \
            --input_format=NCHW \
            --log=error \
            --soc_version=Ascend${chip_name}

        cd ../../
        ```
        参数说明：
        - --model：为ONNX模型文件。
        - --output：输出的OM模型。
        - --framework：5代表ONNX模型。
        - --log：日志级别。
        - --soc_version：处理器型号。
            
        执行成功后生成`models_bs${bs}/clip/clip.om、models_bs${bs}/unet/unet.om`文件。

    3. 执行量化

        运行quant_unet.py脚本进行量化

        ```bash
        # 普通方式
        python3 quant_unet.py \
            --model ${model_base} \
            --model_dir ./models_bs${bs} \
            --prompt_file ./prompts.txt \
            --device 0 \
            --save_path unet_quant

        # 并行方式
        python3 quant_unet.py \
            --model ${model_base} \
            --model_dir ./models_bs${bs} \
            --prompt_file ./prompts.txt \
            --device 0,1 \
            --save_path unet_quant
        ```
        参数说明：
        - --model：模型名称或本地模型目录的路径。
        - --model_dir：存放导出模型的目录。
        - --prompt_file：输入文本文件，按行分割。
        - --save_path：量化模型的储存目录。
        - --device：推理设备ID；可用逗号分割传入两个设备ID，此时会使用并行方式进行推理。

        执行成功后生成`models_bs${bs}/unet_quant`文件夹，包含unet.onnx模型及权重。
