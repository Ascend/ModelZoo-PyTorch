# ResNet50_mmlab_for_pytorch_for_POC 推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

ResNet50是针对移动端专门定制的轻量级卷积神经网络，该网络的batch是24，经过了4个Block，每一个Block中分别有3，4，6，3个Bottleneck，被广泛运用于各种特征提取应用中。


- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmclassification
  commit_id=91b85bb4a5df075ae2690273da32819b298e4395
  model_name=resnet50_b16x8_cifar100
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型     | 大小               | 数据排布格式 |
  |----------|------------------| ------------------------- | ------------ |
  | input    | RGB_INT8 | 24 x 3 x 32 x 32 | NHWC         |


- 输出数据

  | 输出数据 | 大小        | 数据类型 | 数据排布格式 |
  |-----------| -------- | -------- | ------------ |
  | output  | 24 x 100 | FLOAT32  | ND           |



# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------------ |---------| ------------------------------------------------------------ |
| 固件与驱动                                                   | 23.0.rc2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.3.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.13.1   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/open-mmlab/mmclassification.git
   cd mmclassification
   git reset --hard 91b85bb4a5df075ae2690273da32819b298e4395
   pip3 install -v -e .
   cd ..
   git clone https://github.com/open-mmlab/mmdeploy.git
   cd mmdeploy
   git reset --hard b0a350d49e95055136bbef570fd5c635b935c59c
   pip3 install -r requirements.txt
   pip3 install -v -e .
   ```

2.  安装依赖。

    1. 安装基础环境
    ```bash
    pip3 install -r requirements.txt
    ```
    说明：某些库如果通过此方式安装失败，可使用pip单独进行安装。

    2. 安装量化工具

        参考[AMCT(ONNX)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/70RC1alpha001/developmenttools/devtool/atlasamctonnx_16_0004.html)主页安装量化工具。

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   ```
   下载cifar100数据集(http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz),放于resnet50_bs24_for_pytorch_for_POC目录下
   解压缩
   tar -xvf cifar-100-python.tar.gz
   ```
   解压缩后生成文件夹cifar-100-python,推理只使用其中的test文件。

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行**preprocess_resnet50_pytorch.py**脚本，完成预处理
   ```
   python3 preprocess_resnet50_pytorch.py \
        --src_path ./cifar-100-python/test \
        --bin_path ./bin_data 
   ```
   - 参数说明
        - --src_path: 测试数据集地址
        - --bin_path: 生成bin文件地址
   
   运行成功后,同一目录下生成cifar100数据集的可视化数据集pic,bin格式的数据集bin_data以及label文件img_label.txt



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       下载对应的[权重文件](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar100_20210528-67b58a1b.pth)于mmdeploy目录下

   2. 导出onnx文件。

      1. 使用deploy.py导出onnx文件。

         运行deploy.py脚本。

         ```
         cd mmdeploy
         python3 tools/deploy.py ./configs/mmcls/classification_onnxruntime_dynamic.py /usr/local/mmclassification/configs/resnet/resnet50_b16x8_cifar100.py  resnet50_b16x8_cifar100_20210528-67b58a1b.pth /usr/local/mmclassification/demo/demo.JPEG --work-dir ./models/

         ```
         > **说明：**
         生成的end2end.onnx文件位于models目录下.
    3. 模型量化
        在量化前，我们先生成校验数据，以确保量化后模型精度不会损失：
        ```bash
        python3 create_quant_data.py \
            --src_path ./cifar-100-python/test \
            --save_path ./amct_bin_data \
            --amct
        ```
      - 参数说明
        - --src_path: 测试数据集地址
        - --save_path: 生成校验数据bin文件地址
        - --amct: 说明是生成amct量化的校验数据

        然后使用`amct`工具，对ONNX模型进行量化，以进一步提升模型性能：
        ```bash
        amct_onnx calibration \
            --model ./models/end2end.onnx \
            --save_path ./models/resnet50_quant \
            --input_shape "input:24,3,32,32" \
            --data_dir "./amct_bin_data/" \
            --data_types "float32" \
            --calibration_config ./quant.cfg
        ```
        - 参数说明
          - --model: onnx模型
          - --save_path: 保存量化后onnx模型文件地址
          - --input_shape: 模型输入shape
          - --data_dir: 校验数据
          - --data_types: 数据类型
          - --calibration_config: 量化配置文件
        
        量化后的模型存放路径为 `models/resnet50_quant_deploy_model.onnx`。
   
   4. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：**
         该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

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
         atc --framework=5 \
            --model=models/resnet50_quant_deploy_model.onnx \
            --output=models/resnet50_quant_bs24 \
            --input_format=NCHW \
            --input_shape="input:24,3,32,32" \
            --insert_op_conf=./aipp.conf \
            --enable_small_channel=1 \
            --op_select_implmode=high_performance \
            --soc_version=Ascend${chip_name} \
            --log=error
         ```
         - 参数说明：
            -  --model：为ONNX模型文件。
            - --framework：5代表ONNX模型。
            - --output：输出的OM模型。
            - --input_format：输入数据的格式。
            - --input_shape：输入数据的shape。
            - --log：日志级别。
            - --soc_version：处理器型号。
            - --input_format：输入数据的格式。
            - --enable_small_channel:是否使能small_channel优化。
            - --op_select_implmode:高性能模式。
            - --insert_op_conf：使能AIPP，使用该参数后，则输入数据类型为uint8。
         
         运行成功后在models目录下生成**resnet50_quant_bs24.om**模型文件。



2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。


   b.  执行推理。

      ```
      python3 -m ais_bench --model ./models/resnet50_quant_bs24.om --input ./bin_data --output ./ --outfmt TXT --output_dirname dst

      ```

      -   参数说明：

           -   model：需要推理om模型的路径。
           -   input：模型需要的输入bin文件夹路径。
           -   output：推理结果输出路径。
           -   outfmt：输出数据的格式。
           -   output_dirname:推理结果输出子文件夹。


   c.  精度验证。

      调用脚本与数据集标签img_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

      ```
      python3 postprocess_resnet50_pytorch.py  ./dst/  ./img_label.txt ./ result.json
      ```
      - 参数说明

        - ./dst：为生成推理结果所在路径
        - ./img_label.txt：为标签数据
        - ./: 存放结果的路径
        - result.json：为生成结果文件


   d.  性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model ./models/resnet50_quant_bs24.om --loop 100 --batchsize 24

      ```
      - 参数说明
        - --model: om模型
        - --loop: 循环次数
        - --batchsize: 模型batch size


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号  | Batch Size | 数据集  | 精度    | 
|----------|------------|----------|-------|
|  310P3  |       24       | cifar100 | 78.55% |
