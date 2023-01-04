# EfficientNet-b7模型PyTorch离线推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

EfficientNet是图像分类网络，在ImageNet上性能优异，并且在常用迁移学习数据集上达到了相当不错的准确率，参数量也大大减少，说明其具备良好的迁移能力，且能够显著提升模型效果。


- 参考实现：

  ```
  url=https://github.com/rwightman/pytorch-image-models
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 600 x 600 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | --------| -------- | -------- | ------------ |
  | output  | FLOAT32  | batchsize x 1000 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | [CANN推理架构准备](https://www/hiascend.com/software/cann/commercial) |
  | Python                                                       | 3.7.5   | 创建anaconda环境时指定python版本即可，conda create -n ${your_env_name} python==3.7.5 |
  | PyTorch                                                      | 1.11.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/rwightman/pytorch-image-models      # 克隆仓库的代码
   ```

2. 安装依赖，测试环境时可能已经安装其中的一些不同版本的库，故手动测试时不推荐使用该命令安装

   ```
   pip3.7 install -r requirements.txt
   ```

3. 安装开源模型代码仓

   ```
   pip3.7 install efficientNet-pytorch==0.7.1
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

    本模型使用ImageNet 50000张图片的验证集，请参考[ImageNet官网](https://image-net.org/)下载和处理数据集
    
    处理完成后获得分目录的图片验证集文件，目录结构类似如下格式：

    ```text
    imagenet/val/
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ......
    ├── ......
    ```

2. 数据预处理，将原始数据集转换为模型输入的数据，执行preprocess脚本，完成预处理。
   ```
   python3.7 preprocess.py --dataset_path=imagenet/val --save_path=./data_bin
   ```
   - 参数说明：

      -   --dataset_path：数据集路径。
      -   --save_path：预处理后bin文件保存路径。



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       在下一步导出onnx文件时自动下载，若无法下载，请进入链接下载[EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth)，并将权重文件放于${TORCH_HOME}/hub/checkpoints/下，其中TORCH_HOME为PyTorch下载目录

   2. 导出onnx文件。

      1. 使用pth2onnx导出onnx文件。

         运行pth2onnx脚本。

         ```
         python3.7 pth2onnx.py --version=7
         ```

         获得efficientnet_b7_dym_600.onnx文件。

      2. 优化ONNX文件。

         ```
         python3.7 -m onnxsim efficientnet_b7_dym_600.onnx efficientnet_b7_dym_600_sim.onnx --dynamic-input-shape --input-shape 1,3,600,600
         ```

         获得efficientnet_b7_dym_600_sim.onnx文件。

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
          atc --model=efficientnet_b7_dym_600_sim.onnx --framework=5 --input_format=NCHW --input_shape="image:32,3,600,600" --output=efficientnet_b7_32_600_sim --soc_version=Ascend${chip\_name\} --log=debug --optypelist_for_implmode="Sigmoid" --op_select_implmode=high_preformance
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input_format：输入数据的格式。
           -   --input_shape：输入数据的shape。
           -   --soc_version：处理器型号。
           -   --log：日志级别。

           运行成功后生成<u>***efficientnet_b7_32_600_sim.om***</u>模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        mkdir outputs
        python3.7 -m ais_bench --model=efficientnet_b7_32_600_sim.om --input=data_bin --output= outputs --outfmt BIN --device 0  
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --batchsize：批处理大小。
             -   --input：数据预处理后保存文件的路径。
             -   --output：输出文件夹路径。
             -   --outfmt：输出格式（一般为BIN或者TXT）。
             -   --device：NPU的ID，默认填0。

        推理后的输出默认在当前目录参数output创建的输出文件夹下，此处为outputs文件夹。



   3. 精度验证。

      调用脚本与数据集标签label.txt比对，可以获得精度accuracy数据

      ```
       python3.7 postprocess.py --output_dir=outputs/{dir}
      ```

      - 参数说明：

        -   --output_dir：为生成推理结果所在路径。  

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7 -m ais_bench --model=${om_model_path} --loop 5
        ```

      - 参数说明：
        - --model：om模型的路径



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集|  精度 | 性能|
| --------- | -----------| ----------| --------------- |----------------|
| 310P3 |  1       | ImageNet |   84.4 |   62.40      |
| 310P3 |  4       | ImageNet |   84.4 |    72.22      |
| 310P3 |  8       | ImageNet |   84.4 |  73.10     |
| 310P3 |  16       | ImageNet |  84.4 |   66.24      |
| 310P3 |  32       | ImageNet |  84.4 |   75.96      |
| 310P3 |  64       | ImageNet |  84.4 |   75.32      |