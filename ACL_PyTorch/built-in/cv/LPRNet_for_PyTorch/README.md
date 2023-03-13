# LPRNet模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

LPRNet(License Plate Recognition Network)是一个实时的轻量化、高质量、支持可变长车牌的车牌识别网络结构。该结构不需要预先进行车牌字符分割，完全可以端到端的训练，并部署在嵌入式设备上运行。

骨干网络使用原始的RGB图片作为输入，并且计算出大量特征的空间分布。骨干子网络的输出是一个代表对应字符可能性的序列，使用CTC loss对输入与输出序列进行非对齐处理。

- 参考实现：

  ```
  url=https://github.com/sirius-ai/LPRNet_Pytorch.git
  commit_id=7c976664b3f3879efabeaff59c7a117e49d5f29e
  code_path=model/LPRNet.py
  model_name=LPRNet
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 数据类型  | 大小                     | 数据排布格式  |
  | -------- | -------- | ------------------------ | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 24 x 94  | NCHW         |

- 输出数据

  | 输出数据  | 数据类型  | 大小                | 数据排布格式  |
  | -------- | -------- | ------------------- | ------------ |
  | output   | FLOAT32  | batchsize x 68 x 18 | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套        | 版本    | 环境准备指导                                                |
  | ---------- | ------- | ---------------------------------------------------------- |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                        | 6.0.0   | -                                                            |
  | Python                                                      | 3.7.5   | -                                                            |
  | PyTorch                                                     | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

- 该模型需要以下依赖   

  **表 2**  依赖列表

  | 依赖名称               | 版本                    |
  | --------------------- | ----------------------- |
  | torch                 | 1.10.0                  |
  | torchvision           | 0.11.1                  |
  | onnx                  | 1.12.0                  |
  | onnxruntime           | 1.12.1                  |
  | onnx-simplifier       | 0.3.6                   |
  | onnxoptimizer         | 0.3.0                   |
  | onnxsim               | 0.4.8                   |
  | opencv-python         | 4.6.0.66                |
  | Pillow                | 9.0.0                   |
  | numpy                 | 1.21.6                  |
  | tqdm                  | 4.64.0                  |
  | imutils               | 0.5.4                   |
  | protobuf              | 3.20.0                  |
  | auto-optimizer        | 0.1.0                   | 

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/sirius-ai/LPRNet_Pytorch.git
   cd LPRNet_Pytorch
   git reset --hard 7c976664b3f3879efabeaff59c7a117e49d5f29e
   cd ..
   ```

2. 安装依赖。<u>***若有需要编译安装的需补充安装步骤，并使用有序列表说明步骤顺序***</u>

   ```
   pip3 install -r requirements.txt
   ```

   测试环境可能已经安装了其中一些不同版本的依赖库，故手动测试时不推荐使用该命令直接安装。
   
   如果已安装的依赖库版本高于requirements.txt中的推荐版本，可能需要对其进行降级处理。

   > 注意：其中`onnxsim`库在aarch64环境下可能会安装失败，推荐在x86_64环境下进行安装，并对onnx模型进行简化，其他操作可在aarch64环境下运行。

   `auto-optimizer`库[安装参考](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)：

   ```
   git clone https://gitee.com/ascend/msadvisor.git
   cd ./msadvisor/auto-optimizer
   pip3 install -r requirements.txt
   python3 setup.py install
   cd ../../
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型所用测试数据包含在git项目中，路径为 `LPRNet_Pytorch/data/test/`，共1000张图片，对应标签为文件名称。
   
   目录结构如下：

   ```
   data
    └── test
        ├── 京PL3N67.jpg
        ├── 川JK0707.jpg
        ├── 川X90621.jpg
        ├── 沪AMS087.jpg
        ├── ...
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行`LPRNet_preprocess.py`脚本，完成预处理，生成二进制bin数据文件。

   ```
   python3 LPRNet_preprocess.py --img_path=LPRNet_Pytorch/data/test --dst_path=prep_data 
   ```

   参数说明：

   - `--img_path`: 测试数据集路径。
   - `--dst_path`: 生成的bin文件保存路径，默认保存在`./prep_data`路径下。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       PyTorch预训练权重文件保存在git项目 `LPRNet_Pytorch/weights/Final_LPRNet_model.pth` 路径下。

   2. 导出onnx文件。

      1. 使用`LPRNet_pth2onnx.py`导出onnx离线模型文件。

         运行`LPRNet_pth2onnx.py`脚本。

         ```
         python3 LPRNet_pth2onnx.py --pth=LPRNet_Pytorch/weights/Final_LPRNet_model.pth --output=LPRNet.onnx
         ```

         参数说明：

         - `--pth`: 预训练权重文件路径。
         - `--output`: 导出的onnx模型文件保存路径。

         获得LPRNet.onnx文件。

      2. 调整 onnx 模型适配NPU。

         执行 `LPRNet_modify_onnx.py`，使用 auto-optimizer 工具对 onnx 模型进行修改以适配NPU和优化模型。

         ```
         python3 LPRNet_modify_onnx.py --onnx=LPRNet.onnx --output=LPRNet_mod.onnx
         ```

         参数说明：

         - `--onnx`: 要修改的onnx模型路径。
         - `--output`: 修改后onnx模型的保存路径。

         获得LPRNet_mod.onnx文件。

      3. 优化ONNX文件。

         使用 onnxsim 工具对到处的 onnx 模型文件进行常量折叠简化模型。
         因 onnxsim 工具在 aarch64 环境下可能安装失败，推荐在 x86_64 linux 环境下执行该操作：

         ```
         python3 -m onnxsim LPRNet_mod.onnx LPRNet_sim.onnx --input-shape='input:1,3,24,94'
         ```

         获得LPRNet_sim.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         `/usr/local/Ascend/ascend-toolkit/set_env.sh`是CANN软件包安装在默认路径生成的环境变量文件，如自行安装在其他位置，需要改为自己的安装路径。

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
         mkdir models
         atc --framework=5 --model=LPRNet_sim.onnx --input_format=NCHW --input_shape='input:{batchsize},3,24,94' --output=models/LPRNet_bs{batchsize} --log=error --soc_version=Ascend{chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

         - 自定义参数说明：
           - {batchsize}需要指定为要生成的om模型的批处理大小， 如 1、4、8、16等。

         运行成功后生成 LPRNet_bs{batchsize}.om 模型文件，保存在 `models` 路径下。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

      使用 ais_bench 工具对预处理数据进行离线推理，获取对应推理预测结果bin文件。下面以 {batchsize} = 4 为例。

        ```
        mkdir result
        python3 -m ais_bench --model models/LPRNet_bs4.om --input ./prep_data --batchsize 4 --output result --outfmt BIN
        ```

        -   参数说明：

             - --model：om文件路径。
             - --input：预处理文件路径。
             - --batchsize：批处理数量大小。
             - --output：推理结果保存路径。
             - --outfmt：推理结果保存格式，支持 BIN 和 TXT。

        推理后的输出在当前目录`result/{timestamp}`路径下，其中{timestamp}为执行推理认为时的时间戳。

   3. 精度验证。

      调用`LPRNet_postprocess.py`脚本与数据集标签比对，可以获得Accuracy数据。

      ```
      python3 LPRNet_postprocess.py ./result/{timestamp}
      ```

      - 参数说明：

        - result/{timestamp}：为生成推理结果所在路径，请将{timestamp}替换为实际路径名称。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
        python3 -m ais_bench --model=${om_model_path} --loop=1000 --batchsize=${batch_size}
        ```

      - 参数说明：

        - --model：om模型路径。
        - --batchsize：批处理数量大小。
        - --loop：循环执行的次数。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能与精度参考下列数据。

| 芯片型号     | Batch Size | 数据集 | 精度        | 性能             |
| ----------- | :--------: | :---:  | :------:   | :--------------: |
| Ascend310P3 |  1         |  -     |  88.3%     |   5714.29 fps    |
| Ascend310P3 |  4         |  -     |  89.6%     |   14981.27 fps   |
| Ascend310P3 |  8         |  -     |  89.5%     |   20512.82 fps   |
| Ascend310P3 |  16        |  -     |  90.2%     |   25039.12 fps   |
| Ascend310P3 |  32        |  -     |  90.2%     |   27313.22 fps   |
| Ascend310P3 |  64        |  -     |  89.8%     |   20043.85 fps   |
