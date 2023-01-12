# ENet 模型-推理指导


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

ENet是用于实时语义分割的深度神经网络体系结构，该网络借鉴ResNet的残Bottleneck结构来组建网络，其五个主要阶段中，前三个为编码器，后两个为解码器，形成的网络结构在满足快速推理的情况下，仍旧能够达到较高的精度。


- 参考实现：

  ```
  url=https://github.com/Tramac/awesome-semantic-segmentation-pytorch
  commit_id=5843f75215dadc5d734155a238b425a753a665d9
  model_name=contrib/cv/segmentation/ENet
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 480x 480 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output  | FLOAT32  | batchsize x 19 x 96 x 96 | NCHW           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

  该模型使用Cityscapes数据集作为训练集，其下的val中的500张图片作为验证集。推理部分只需要用到这500张验证图片，验证集输入图片存放在`./datasets/citys/leftImg8bit/val`，验证集target存放在`./datasets/citys/gtFine/val`：

   ```
   datasets
   ├── citys
     ├── gtFine
           ├── val
     ├── leftImg8bit
           ├── val  
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行`ENet_preprocess.py`脚本，完成预处理。

   ```
   Python3.7 ENet_preprocess.py --src_path ./datasets/citys --save_path ./prep_dataset
   ```
   - 参数说明
      - src_path: 原始数据路径
      - save_path：保存结果路径
  

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       ```
       wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Enet/PTH/enet_citys.pth
       ```

   2. 导出onnx文件。

      1. 使用`ENet_pth2onnx.py`导出onnx文件。
         运行`ENet_pth2onnx.py`脚本。

         ```
         Python3.7 ENet_pth2onnx.py --input-file enet_citys.pth --output-file enet_citys.onnx
         ```
         - 参数说明
            - input-file： 输入pth文件
            - output-file： 输出onnx文件

         获得`enet_citys.onnx`文件。


   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         会显如下：
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
            --model=./enet_citys.onnx \
            --output=./enet_citys_bs1 \
            --input_format=NCHW \
            --input_shape="image:1,3,480,480" \
            --log=error \
            --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成`enet_citys_bs1.om`模型文件。

2. 开始推理验证。
   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais-bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

        ```
        python3 -m ais_bench  --model ./enet_citys_bs1.om \
               --input ./prep_dataset/ \
               --output ./ \
               --outfmt BIN \
               --output_dirname result
        ```

        -   参数说明：

             -   model：om模型
             -   input：输入文件
             -   output：输出路径
             -   outfmt: 输出格式
             -   output_dirname：输出文件夹
                  	...

        推理后的输出默认在当前目录`result`下。

        >**说明：** 
        >执行ais-bench工具请选择与运行环境架构相同的命令

   3. 精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

      ```
       python3.7 ENet_postprocess.py --src_path=./datasets/citys  --result_dir ./result_summary.json | tee eval_log.txt
      ```
      - 参数说明：
        - src_path：原数据路径  
        - result_dir：结果json路径

      结果保存在`eval_log.txt`
   4. 性能验证。

      可使用ais-bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型
        - --batchsize：模型batchsize
        - --loop: 循环次数



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度 | 性能 |
| -------- | ---------- | ------ | ---- | ---- |
|    310P3      |       1     |     citys   |   mIoU:54.11   |    1071  |
|    310P3      |       4     |     citys   |   mIoU:54.11   |    1327  |
|    310P3      |       8     |     citys   |   mIoU:54.11   |    1224  |
|    310P3      |       16     |     citys   |   mIoU:54.11   |    1205  |
|    310P3      |       32     |     citys   |   mIoU:54.11   |    1205  |
|    310P3      |       64     |     citys   |   mIoU:54.11   |    1009  |