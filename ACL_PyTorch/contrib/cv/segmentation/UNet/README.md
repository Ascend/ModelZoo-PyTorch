# Unet模型-推理指导

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

UNet是由FCN改进而来的图像分割模型，其网络结构像U型，分为特征提取部分和上采样特征融合部分。

- 参考实现：

  ```
  url=https://github.com/milesial/Pytorch-UNet
  commit_id=6aa14cbbc445672d97190fec06d5568a0a004740
  model_name=UNet
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型    | 大小                        | 数据排布格式 |
  |---------|---------------------------| ------------------------- | ------------ |
  | input    | FLOAT32 | batchsize x 3 x 572 x 572 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- |---------------------------|--------| ------------ |
  | output   | FLOAT32  | batchsize x 3 x 388 x 388 | NCHW   |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本      | 环境准备指导                                                 |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.4  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.3.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.7.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/milesial/Pytorch-UNet.git
   cd Pytorch-UNet
   git reset --hard 6aa14cb
   cd ..
   mv ./Pytorch-UNet ./Pytorch_UNet
   ```
   
2. 安装依赖

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持carvana数据集，[下载链接](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data)。数据集train.zip以及train_masks.zip分别作为训练和标签文件，上传并解压到源码包路径下，目录结构如下：

   ```
   carvana
   ├── train
   └── train_masks 
   ```
   
2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行preprocess_unet_pth.py脚本，将原始数据（.jpg）转化为二进制文件（.bin）。
   ```
   python3 preprocess_unet_pth.py --src_path=./carvana/train --save_bin_path=./prep_bin
   ```
   
    - 参数说明：
      - --src_path：原始数据集所在路径。
      - --save_bin_path：输出的二进制文件所在路径。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用Pytorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      UNet.pth权重文件[下载链接](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Unet/PTH/UNet.pth)。

   2. 导出onnx文件。

      1. 移动unet_pth2onnx.py至Pytorch_Unet目录，使用unet_pth2onnx.py导出onnx文件。

         ```
         mv unet_pth2onnx.py ./Pytorch_UNet
         python3 ./Pytorch_UNet/unet_pth2onnx.py ./UNet.pth ./UNet_dynamic_bs.onnx
         ```
         
         获得UNet_dynamic_bs.onnx文件。

      2. 使用onnxsim精简onnx文件。
         ```
         python3 -m onnxsim --dynamic-input-shape --input-shape="1,3,572,572" UNet_dynamic_bs.onnx UNet_dynamic_sim.onnx
         ```
         获得UNet_dynamic_sim.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
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

      3. 执行ATC命令。

         ```
         atc --model=UNet_dynamic_sim.onnx --framework=5 --output=UNet_bs${batch_size} --input_format=NCHW --input_shape='actual_input_1:${batch_size},3,572,572' --log=info --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***UNet_bs${batch_size}.om***</u>模型文件。

2. 开始推理验证

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

      ```
      python3 -m ais_bench --model=UNet_bs${batch_size}.om --input=./prep_bin --output=result --output_dirname=bs${batch_size} --batchsize=${batch_size}
      ```

      - 参数说明：

        -   --model：om文件路径。
        -   --input：输入数据目录。
        -   --output：推理结果输出路径。
        -   --output_dirname: 推理结果输出目录。

   3. 精度验证。

      1. 处理summary.json文件，依据json文件信息更改推理输出文件名称。

         ```
         python3 json_parse.py --output=result/bs${batch_size}/
         ```

         - 参数说明：

           - --output：推理结果生成路径。
      
      2. 调用postprocess_unet_pth.py脚本与train_masks标签数据比对，可以获得Accuracy数据。

         ```
         python3 postprocess_unet_pth.py --output=result/bs${batch_size} --label=./carvana/train_masks --result=./result.txt
         ```

         - 参数说明：

           - --output：推理结果生成路径。
           - --label：标签文件路径。
           - --result：生成IOU结果文件。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
        python3 -m ais_bench --model=UNet_bs${batch_size}.om --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om文件路径。
        - --batchsize：batch大小

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| Batch Size | 数据集  | 精度         | 310P3       | 310B1     |
| ---------- | ------- | ------------ | ----------- | --------- |
| 1          | carvana | IOU:0.986305 | 75.0603     | 12.78     |
| 4          | carvana | IOU:0.986305 | 71.2920     | 11.75     |
| 8          | carvana | IOU:0.986305 | 68.5334     | 11.56     |
| 16         | carvana | IOU:0.986305 | 67.7102     | 11.29     |
| 32         | carvana | IOU:0.986305 | 65.5027     | 11.18     |
| 64         | carvana | IOU:0.986305 | 49.0184     | NA        |
|            |         | **最优性能** | **75.0603** | **12.78** |