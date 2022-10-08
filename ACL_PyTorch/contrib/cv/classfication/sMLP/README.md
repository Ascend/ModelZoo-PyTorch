# sMLP模型-推理指导

- [概述](#概述)
- [推理环境](#推理环境)
- [快速上手](#快速上手)
    - [获取源码](#获取源码)
    - [准备数据集](#准备数据集)
    - [模型转换](#模型转换)
    - [推理验证](#推理验证)
- [精度&性能](#精度性能)


----
# 概述

随着 Transformer 在计算机视觉领域内雨后春笋般地涌现，作者探讨了 Transformer 的核心 Self-Attention 是否为在图像识别领域取得优异表现的关键因素。为此，作者基于现有的 MLP-based 视觉模型构建了 sMLPNet 无注意力网络。具体来说，作者将用于 token 混合的 MLP 模块替换为新颖的稀疏 MLP (sMLP) 模块。对于二维图像 token，sMLP 沿横向或纵向应用一维 MLP，参数在行或列之间共享。通过稀疏连接和权重共享，sMLP 模块显著降低了模型参数量和计算复杂度，避免了常见于 MLP 这一类模型的过拟合问题。在 ImageNet-1K 数据集上进行训练，仅有 24M 参数量的 sMLPNet，其 top-1 准确率高达 81.9%，表现优于大多数同规模的 CNN 和视觉 Transformer。当参数扩展到 66M 时，sMLPNet 的 top-1 准确率提升至 83.4%，与当前表现最优的 Swin Transformer 相当。sMLPNet 的成功表明 Self-Attention 不一定是计算机视觉领域内的灵丹妙药。

- 论文  
    [Sparse MLP for Image Recognition: Is Self-Attention Really Necessary?](https://arxiv.org/abs/2109.05422)  
    [Chuanxin Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+C), [Yucheng Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+Y), [Guangting Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+G), [Chong Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+C), [Wenxuan Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+W), [Wenjun Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+W)

- 参考实现：

    ```
    url = https://github.com/microsoft/SPACH.git
    branch = main
    commit_id = b11b098970978677b7d33cc3424970152462032d
    model_name = sMLPNet-T
    ```

- 输入数据

    | 输入数据 | 数据类型 | 数据排布格式 | 大小          |
    | -------- | -------- | ---------- | ------------ |
    | input    | RGB_FP32 | NCHW       | batchsize x 3 x 224 x 224 |


- 输出数据

    | 输出数据 | 数据类型 | 数据排布格式 | 大小       |
    | ------- | -------- | ---------- | ---------- |
    | output  | FLOAT32  | ND         | batchsize x 1000|


----
# 推理环境

- 该模型离线推理使用 Atlas 300I Pro 推理卡，推理所需配套的软件如下：

    | 配套      | 版本    | 环境准备指导 |
    | --------- | ------- | ---------- |
    | firmware | 1.82.22.2.220 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | driver | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | CANN      | 5.1.RC2 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
    | Python    | 3.7.5   | -          |
    | PyTorch   | 1.5.0   | -          |
    

----
# 快速上手

## 获取源码

1. 获取源码
    ```shell
    git clone https://github.com/microsoft/SPACH.git
    cd SPACH
    git checkout main
    git reset b11b098970978677b7d33cc3424970152462032d --hard
    ```
    说明：本仓代码会引用源码仓代码，需要在源码仓SPACH目录下执行前后处理与转ONNX的脚本。

2. 安装依赖

    ```shell
    conda create -n spach python=3.7.5
    conda activate spach
    pip3 install -r requirements.txt
    ```

## 准备数据集

1. 获取原始数据集
    本离线推理项目使用 ILSVRC2012 数据集（ImageNet-1k）的验证集进行精度验证。从 [http://image-net.org/](http://image-net.org/) 下载数据集并解压， val 目录结构遵循 [torchvision.datasets.ImageFolder](https://gitee.com/link?target=https%3A%2F%2Fpytorch.org%2Fvision%2Fstable%2Fgenerated%2Ftorchvision.datasets.ImageFolder.html%23torchvision.datasets.ImageFolder) 的标准格式：
    ```
    /path/to/imagenet/
    ├──val/
    │  ├── n01440764
    │  │   ├── ILSVRC2012_val_00000293.JPEG
    │  │   ├── ILSVRC2012_val_00002138.JPEG
    │  │   ├── ...
    │  ├── ...
    ```

2. 数据预处理

   源码仓目录下执行数据预处理脚本，将原始数据集中的 JPEG 图片转换为模型输入需要的 bin 文件。
    ```shell
    python3.7 smlp_preprocess.py --save_dir ./imagenet-val-bin --data_root /opt/npu/imagenet/
    ```
    参数说明：  
    + --data_root: 数据集路径  
    + --save_dir: 存放预处理生成的 bin 文件的目录路径  

    运行成功后，每张原始 JPEG 图片对应生成一个二进制 bin 文件，存放于源码仓目录下的 imagenet-val-bin 目录内。


## 模型转换

使用 PyTorch 将模型权重文件（.pth）转换为 ONNX 模型（.onnx），再使用 ATC 工具将 ONNX 模型转为离线推理模型（.om）。

1. 获取权重文件  
    由于开源仓未提供预训练 pth，所以本推理项目使用自己训练的 [pth 权重文件](https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/smlp_t.pth)，将其下载到开源仓目录下。
    ```shell
    wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/sMLP/smlp_t.pth
    ```

2. 导出ONNX文件

    在开源仓目录下执行前处理脚本加载权重文件（.pth）并将其转换为 ONNX 模型
    ```shell
    python smlp_pth2onnx.py --model_name smlpnet_tiny  --pth_path ./smlp_t.pth --onnx_path ./sMLPNet-T.onnx --opset_version 11
    ```
    参数说明：
    + --model_name: 模型名称
    + --pth_path: PyTorh预训练权重文件路径
    + --onnx_path: ONNX模型路径
    + --opset_version: ONNX算子集版本，默认11

    运行成功后，在开源仓目录下将生成 sMLPNet-T.onnx 文件。

3. 使用ATC工具将ONNX模型转OM模型。
    
    step1: 查看NPU芯片名称（$\{chip\_name\}）

    ```shell
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
    step2: ONNX模型转OM模型  
    在开源仓目录下执行以下命令，将 ONNX 模型转为 OM 模型

    ```shell
    # 配置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
    # 执行 ATC 进行模型转换
    atc --framework=5 \
        --model=sMLPNet-T.onnx \
        --output=sMLPNet-T-batch8-high \
        --input_format=NCHW \
        --input_shape="input:8,3,224,224" \
        --log=error \
        --soc_version=Ascend${chip_name} \
        --op_precision_mode=op_precision.ini
    ```
    atc 命令参数说明：  
    + --framework: 5代表ONNX模型。  
    + --model: ONNX模型路径。  
    + --input_format: 输入数据的排布格式。  
    + --input_shape: 输入数据的shape。  
    + --output: 输出的OM模型的保存路径。  
    + --log: 日志级别。  
    + --soc_version: 处理器型号。  
    + --op_precision_mode: 选择算子的实现模式：高性能/高精度。

    运行成功后生成 sMLPNet-T-batch8-high.om 模型文件。


## 推理验证

1. 准备推理工具

    推理工具使用ais_infer，须自己拉取源码，打包并安装。
    ```shell
    # 指定CANN包的安装路径
    export CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest
    # 获取源码
    git clone https://gitee.com/ascend/tools.git
    cd tools/ais-bench_workload/tool/ais_infer/backend/
    # 打包
    pip3 wheel ./   # 会在当前目录下生成 aclruntime-xxx.whl，具体文件名因平台架构而异
    # 安装
    pip3 install --force-reinstall aclruntime-xxx.whl
    ```
    参考：[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer#%E4%BB%8B%E7%BB%8D)

2. 离线推理

    进入 ais_infer.py 所在目录并执行以下命令对预处理数据推理，将推理结果保存在开源仓目录下。
    ```shell
    # 设置环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # 创建目录存放推理结果
    mkdir </path/to>/SPACH/infer_results/

    # 执行推理
    python ais_infer.py  \
        --model </path/to>/SPACH/sMLPNet-T-batch8-high.om \
        --input </path/to>/SPACH/imagenet-val-bin \
        --output </path/to>/SPACH/infer_results/ \
        --outfmt NPY \
        -–batchsize 8
    ```
    ais_infer 参数说明:
    + --model: OM模型路径
    + --input: 存放预处理bin文件的目录路径
    + --output: 存放推理结果的目录路径
    + --outfmt: 推理输出文件的格式
    + -–batchsize: 批处理大小

    运行成功后，在--output指定的目录下，会生成一个根据执行开始时间来命名的子目录，用于存放推理结果文件。

3. 精度验证

    推理结束后，回到开源仓目录，执行后处理脚本计算模型在 Top@1 与 Top@5 上的准确率。
    ```shell
    python smlp_postproces.py --infer_result_dir SPACH/infer_results/2022_07_09-18_05_40/
    ```
    参数说明:
    + --infer_result_dir: 存放推理结果的目录路径。例如本例中为SPACH/infer_results/2022_07_09-18_05_40/

    输出如下：
    ```
    acc@1:0.8125, acc@5:0.9549
    ```    

4. 性能验证

    进入 ais_infer.py 所在目录，纯推理100次，然后通过日志获取模型的性能指标。
    ```shell
    cd ais_infer/
    mkdir tmp_out   # 提前创建临时目录用于存放纯推理输出
    python3.7 ais_infer.py --model /path/to/model --output ./tmp_out --outfmt BIN  --batchsize ${bs} --loop 100
    rm -r tmp_out   # 删除临时目录
    ```
    说明：
    1. **性能测试前使用`npu-smi info`命令查看 NPU 设备的状态，确认空闲后再进行测试。否则测出来性能会低于模型真实性能。**
    2. 运行结束后，日志中 **Performance Summary** 一栏会记录性能相关指标，找到以关键字 **throughput** 开头的一行，行尾的数字即为 OM 模型的吞吐率。


----
# 精度&性能

1. OM 离线推理的 top1 准确率为 **81.25%**，与开源仓上公布的 top1 准确率相比，精度下降在 1% 以内。

    | NPU实测精度     | 开源仓精度      | 相对误差       |
    | -------------- | -------------- | -------------- |
    | acc@1 = 81.25% | [acc@1 = 81.9%](https://github.com/microsoft/SPACH#main-results-on-imagenet-with-pretrained-models) | 0.79% |

2. 在 310P 设备上，当 batchsize 为 **8** 时模型性能最优，吞吐率达 **298.7** fps，是T4设备最优性能的 0.81 倍。

    | batchsize | 310P性能 | T4性能 | 310P/T4 |
    | --------- | -------- | ------ | ------ |
    | 1         | 171.6    | 177.7  | 0.97   |
    | 4         | 273.5    | 341.5  | 0.80   |
    | 8         | 298.7    | 359.0  | 0.83   |
    | 16        | 290.0    | 363.7  | 0.80   |
    | 32        | 273.0    | 371.0  | 0.74   |
    | 64        | 257.5    | 359.1  | 0.72   |
    | 性能最优bs | 298.7    | 371.0  | 0.81   |
    
    注：已通过性能评审。

