# UNet++ (Nested UNet)模型-推理指导


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

UNet++由不同深度的U-Net组成，其解码器通过重新设计的跳接以相同的分辨率密集连接，主要用于医学图像分割任务。

- 参考论文：

  [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165)


- 参考实现：

  ```
  url=https://github.com/4uiiurz1/pytorch-nested-unet
  branch=master
  commit_id=557ea02f0b5d45ec171aae2282d2cd21562a633e
  ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据       | 数据类型 | 大小                    | 数据排布格式 |
  | -------------- | -------- | ----------------------- | ------------ |
  | actual_input_1 | RGB_FP32 | batchsize x 3 x 96 x 96 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                    | 数据排布格式 |
  | -------- | -------- | ----------------------- | ------------ |
  | output_1 | RGB_FP32 | batchsize x 1 x 96 x 96 | NCHW         |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.12.1  | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码。

2. 在同级目录下，下载第三方源码并打补丁。

   ```
   git clone https://github.com/4uiiurz1/pytorch-nested-unet
   cd pytorch-nested-unet
   git reset --hard 557ea02f0b5d45ec171aae2282d2cd21562a633e
   patch -p1 < ../nested_unet.diff
   cd ..
   ```

3. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型使用[2018 Data Science Bowl数据集](https://gitee.com/link?target=https%3A%2F%2Fwww.kaggle.com%2Fc%2Fdata-science-bowl-2018)进行推理测试 ，用户自行获取 `stage1_train.zip` 后，将文件解压并上传数据集到第三方源码的 `inputs/data-science-bowl-2018` 目录下。数据集及第三方源码的目录结构关系如下所示：

   ```
   pytorch-nested-unet/
   |-- LICENSE
   |-- README.md
   |-- archs.py
   |-- dataset.py
   |-- inputs
   |   `-- data-science-bowl-2018 
   |       `-- stage1_train # 解压后数据集
   |			|-- xxx
   |			|-- yyy
   |   		`-- ...
   |-- ...
   |-- preprocess_dsb2018.py # 数据集预处理脚本
   |-- ...
   `-- val_ids.txt
   ```

2. 执行原代码仓提供的数据集预处理脚本，生成处理后的数据集文件夹dsb2018_96。

   ```
   cd pytorch-nested-unet
   python3 preprocess_dsb2018.py
   cd ..
   ```
   
3. 将第2步得到的数据集转换为模型的输入数据。

   执行 nested_unet_preprocess.py 脚本，完成数据预处理。

   ```
   python3 nested_unet_preprocess.py ./pytorch-nested-unet/inputs/dsb2018_96/images ${prep_data} ./pytorch-nested-unet/val_ids.txt
   ```
   参数说明：

   - --参数1：原数据集所在路径。
   - --参数2：生成数据集的路径。
   - --参数3：验证集图像id文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       获取权重文件[nested_unet](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Unet%2B%2B/PTH/nested_unet.pth)

   2. 导出onnx文件。

      1. 使用nested_unet_pth2onnx.py导出动态batch的onnx文件。

         ```
         python3 nested_unet_pth2onnx.py ${pth_file} ${onnx_file}
         ```

         参数说明：

         - --pth_file：权重文件。
         - --onnx_file：生成 onnx 文件。
      
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
          # bs = [1, 4, 8, 16, 32, 64]
         atc --framework=5 --model=./nested_unet.onnx --input_format=NCHW --input_shape="actual_input_1:${bs},3,96,96" --output=nested_unet_bs${bs} --log=error --soc_version=Ascend${chip_name}
         ```
      
         运行成功后生成nested_unet_bs${bs}.om模型文件。
      
         参数说明：
         - --model：为ONNX模型文件。
         - --framework：5代表ONNX模型。
         - --output：输出的OM模型。
         - --input\_format：输入数据的格式。
         - --input\_shape：输入数据的shape。
         - --log：日志级别。
         - --soc\_version：处理器型号。
   
2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```
      python3 -m ais_bench --model=nested_unet_bs${bs}.om  --batchsize=${bs} \
      --input ${prep_data} --output result --output_dirname result_bs${bs} --outfmt BIN
      ```
      
      参数说明：
      
      -   --model：om模型路径。
      -   --batchsize：批次大小。
      -   --input：输入数据所在路径。
      -   --output：推理结果输出路径。
      -   --output_dirname：推理结果输出子文件夹。
      -   --outfmt：推理结果输出格式
   
3. 精度验证。
  
      调用脚本与真值比对，可以获得精度结果。
   
      ```
    python3 nested_unet_postprocess.py ./result/result_bs${bs} ./pytorch-nested-unet/inputs/dsb2018_96/masks/0/
    ```

      参数说明：
   
      - --参数1：推理输出目录。
      - --参数2：真值所在目录。
   
4. 可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
  
      ```
      python3 -m ais_bench --model=nested_unet_bs${bs}.om --loop=50 --batchsize=${bs}
      ```
      
      参数说明：
      - --model：om模型路径。
      - --loop：推理单组数据的循环次数。
      - --batchsize：批次大小。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，UNet++模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集                 | 开源精度（IoU） | 精度指标（IoU） |
| ----------- | ---------- | ---------------------- | --------------- | --------------- |
| Ascend310P3 | 16         | data-science-bowl-2018 | 0.842           | 0.838           |

| 芯片型号    | Batch Size | 性能指标（FPS） |
| ----------- | ---------- | --------------- |
| Ascend310P3 | 1          | 1565.07         |
| Ascend310P3 | 4          | 2485.11         |
| Ascend310P3 | 8          | 2526.85         |
| Ascend310P3 | 16         | 2483.1          |
| Ascend310P3 | 32         | 2217.36         |
| Ascend310P3 | 64         | 1760.00         |
