# CenterNet模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

  - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

CenterNet 是在 2019 年提出的用于目标检测的模型，相比传统依靠 anchors的检测网络，CenterNet 是一种 anchor-free 的目标检测网络，其输出主要为heatmap，获取该热力图分布的中心点即作为目标的中心点。而目标的其他输出，如尺寸和偏移量等则通过在特征图中通过回归得到，这种方法原理简单，兼容性强，在速度和精度上都比较有优势。


- 论文：

  [Objects as Points: Xingyi Zhou, Dequan Wang, Philipp Krähenbühl.(2019)](https://arxiv.org/abs/1904.07850)

- 参考实现：

  ```
  url= https://github.com/xingyizhou/CenterNet 
  branch=master 
  commit_id=2b7692c377c6686fb35e473dac2de6105eed62c6
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 512 x 512 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                       | 数据排布格式 |
  | -------- | -------- | -------------------------- | ------------ |
  | output1  | FLOAT32  | batchsize x 80 x 128 x  128 | ND           |
  | output2  | FLOAT32  | batchsize x 2 x 128 x  128 | ND           |
  | output3  | FLOAT32  | batchsize x 2 x 128 x  128 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套       | 版本    | 环境准备指导                                                 |
  | ---------- | ------- | ------------------------------------------------------------ |
  | 固件与驱动 | 23.0.rc1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN       | 7.0.RC1.alpha003 | -                                                    |
  | Python     | 3.9.11   | -                                                            |
  | torch_aie  | 6.3.rc2  | - |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。
   ```
   pip install -r requirements.txt
   ```
   
2. 获取源码并安装。

   ```
    # 获取CenterNet主仓工程，并编译相关依赖
    git clone https://github.com/xingyizhou/CenterNet
    cd CenterNet && git reset --hard 4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c
    cd ./src/lib/models/networks
    rm -r DCNv2
    rm -r pose_dla_dcn.py

    # 在CenterNet目录中获取DCNv2_latest主仓工程
    git clone https://github.com/jinfagang/DCNv2_latest.git
    mv DCNv2_latest DCNv2
    cd DCNv2 && git reset --hard 864c2850a780979f817c126b0e1a363a941ce07b
    rm dcn_v2.py && cd ../../../../../../

    # 替换关键文件
    mv dcn_v2.py CenterNet/src/lib/models/networks/DCNv2
    mv pose_dla_dcn.py DCNv2.patch CenterNet/src/lib/models/networks

    # 编译外部依赖
    cd CenterNet/src/lib/external
    make
    cd ../models/networks
    patch -p0 < DCNv2.patch

    # 进入DCNv2目录安装该库
    cd ./DCNv2
    python setup.py build develop
    cd ../../../../../../
    export PYTHONPATH=./CenterNet/src/:$PYTHONPATH

   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   获取COCO数据集：coco2017，下载其中val2017图片及其标注文件（2017 Val images，2017 Train/Val annotations），放入/data/datasets/coco路径下，val2017目录存放coco数据集的验证集图片，“annotations”目录存放coco数据集的“instances_val2017.json”。目录结构如下：

   ```
   data
   ├── datasets
   │   ├── coco
   │   │   ├── annotations
   │   │   ├── val2017
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行CenterNet_preprocess.py脚本，完成预处理。

   ```
   python CenterNet_preprocess.py /data/datasets/coco/val2017 prep_dataset
   ```

   参数说明：

   - /data/datasets/coco/val2017:  原始数据验证集所在路径。
   - prep_dataset:   输出的二进制文件保存路径。

   运行成功后，生成“prep_dataset”文件夹，prep_dataset目录下生成的是供模型推理的bin文件。


## 模型推理<a name="section741711594517"></a>

**使用PyTorch JIT.trace()工具将模型权重文件.pth转换为torchscript文件**

1. 获取权重文件。放在当前目录下 [ctdet_coco_dla_2x.pth](https://pan.baidu.com/s/1e8TIeBvWzb15UEHWCDZcSQ )
    提取码：d446

2. 导出torchscript文件

    使用ctdet_coco_dla_2x.pth导出torchscript文件。

    在CenterNet根目录下，运行CenterNet_pth2script.py脚本。

    ```
    python CenterNet_pth2torchscript.py ctdet_coco_dla_2x.pth CenterNet_torchscript.pt
    ```

    获得CenterNet_torchscript.pt文件。

**对JIT.trace()生成的TorchScript模型执行torch_aie编译，导出Ascend NPU支持的AIE_TorchScript模型**

1. 配置环境变量。

    ```
    # 配置CANN toolkit环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    # [若使用推理引擎安装包] 配置环境中推理引擎安装路径下的环境变量设置脚本
    source {AIE_PATH}/set_env.sh

    # [若使用PT框架工程调用AIE工程的.so编译安装] 需设置ASCENDIE_HOME路径环境变量
    export ASCENDIE_HOME={AIE_PATH}
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

3. 更换torch, torchvision配套版本至推理引擎PyTorch框架插件支持的Torch版本

    ```shell
    pip install torch==2.0.1 torchvision==0.15.0
    ```

4. 执行Torch-AIE编译，导出Ascend NPU支持的AIE_TorchScript模型

    ```shell
    soc_version="Ascend310P3" # 通过npu-smi info获得
    python CenterNet_export_torch_aie.py               \
        --torch-script-path ./CenterNet_torchscript.pt \
        --batch-size 1                                 \
        --save-path ./                                 \
        --soc-version ${soc_version}
    ```

    - 参数说明：
    -   --model：为trace生成的TorchScript模型文件
    -   --batch-size：模型的BatchSize
    -   --save-path: 编译生成AIE_TorchScript模型的保存路径
    -   --soc-version：处理器型号。

    运行成功后生成 CenterNet_torchscriptb1_torch_aie.pt 模型文件。

**开始推理验证**

1. 执行推理。

    ```shell
    python CenterNet_inference.py                                \
        --aie-module-path ./CenterNet_torchscriptb1_torch_aie.pt \
        --batch-size 1                                           \
        --processed-dataset-path ./prep_dataset/                 \
        --output-save-path ./result_aie/                         \
        --device-id 0
    ```

    - 参数说明：
        -   --aie-module-path: 编译生成AIE_TorchScript模型的路径
        -   --batch-size：模型的BatchSize
        -   --processed-dataset-path：COCO数据集经过预处理后的.bin目录
        -   --output-save-path: 推理生成.bin文件保存目录
        -   --device-id: Ascend NPU ID(可通过npu-smi info查看)

    运行成功后将打印该模型在Ascend NPU推理结果的性能信息

3. 精度验证。

    在CenterNet根目录下，执行脚本CenterNet_postprocess_s1.py，将AIE_TorchScript模型生成的.bin结果转换为可校验的文件格式

    ```
    python CenterNet_postprocess_s1.py \
        --bin-data-path=./result_aie/  \
        --dataset=/data/datasets       \
        --save-dir=./postprocessed
    ```

    然后执行执行脚本CenterNet_postprocess_s2.py 获得模型精度信息

    ```
    python CenterNet_postprocess_s2.py      \
        --dataset=/data/datasets            \
        --postprocessed_dir=./postprocessed
    ```

    - 参数说明：
        - --bin-data-path：AIE_TorchScript模型推理结果文件路径
        - --dataset: COCO原始数据集路径
        - --save-dir: 后处理过文件的保存路径
        - --postprocessed_dir: 后处理过文件的保存路径

    运行成功后将打印COCO2017数据集使用CenterNet模型在Ascend NPU推理结果的精度信息

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

基于推理引擎及Ascend NPU完成推理计算，精度与性能参考下列数据：

| Batch Size | 数据集   | 精度 | 310P3 |
| ---------- | -------- | ---- | ------- |
| 1          | COCO2017 | 36.5 | 22.18 fps |
| 4          | COCO2017 | 36.5 | 23.48 fps |
| 8          | COCO2017 | 36.5 | 22.77 fps |
| 16         | COCO2017 | 36.5 | 22.54 fps |
| 32         | COCO2017 | 36.5 | 21.89 fps |
|  |  | **最优性能** | **23.48** |


# 公网地址说明
代码涉及公网地址参考 public_address_statement.md
