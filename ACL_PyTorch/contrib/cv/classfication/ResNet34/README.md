# ResNet34 模型-推理指导


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

Resnet是残差网络(Residual Network)的缩写,该系列网络广泛用于目标分类等领域以及作为计算机视觉任务主干经典神经网络的一部分，典型的网络有resnet50, resnet34等。Resnet34网络参数量更少，更易训练。


- 参考实现：

  ```
  url=https://github.com/pytorch/vision
  commit_id=7d955df73fe0e9b47f7d6c77c699324b256fc41f
  model_name=contrib/cv/classfication/ResNet34
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | 1 x 1000 | ND           |




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

1. 获取源码。

   ```
   git clone https://github.com/pytorch/vision
   cd vision
   python3.7 setup.py install
   cd ..
   ```

2. 安装依赖

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型使用ImageNet 50000张图片的验证集，请前往ImageNet官网下载数据集

    ```
    ├── ImageNet
    |   ├── val
    |   |    ├── ILSVRC2012_val_00000001.JPEG
    │   |    ├── ILSVRC2012_val_00000002.JPEG
    │   |    ├── ......
    |   ├── val_label.txt
    ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行imagenet_torch_preprocess.py脚本，完成预处理
   ```
   python3.7 imagenet_torch_preprocess.py resnet ./Imagenet/val ./prep_dataset
   ```



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       ```
       wget https://download.pytorch.org/models/resnet34-b627a593.pth
       ```

   2. 导出onnx文件。

      1. 使用resnet34_pth2onnx.py导出onnx文件

         运行resnet34_pth2onnx.py脚本。

         ```
         python3.7 resnet34_pth2onnx.py ./resnet34-b627a593.pth resnet34.onnx
         ```

         获得`resnet34.onnx`文件。


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
         atc --framework=5 --model=./resnet34.onnx --output=resnet34_bs1 --input_format=NCHW --input_shape="image:1,3,224,224" --log=error --soc_version=Ascend${chip_name} 
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
          

           运行成功后生成`resnet34_bs1.om`模型文件。

2. 开始推理验证

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。 

   2. 执行推理。

        ```
        python3 -m ais_bench --model resnet34_bs1.om \
					--input prep_dataset \
					--output ./ \
					--output_dirname result \
					--outfmt TXT
        ```

        -   参数说明：

             -   model：om模型
             -   input：推理输入数据
             -   output：推理结果输出路径
             -   output_dirname: 推理结果文件夹
             -   outfmt: 推理结果格式
                  	

        推理后的输出默认在当前目录result下。


   3. 精度验证。

      调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。
      ```
      python3.7 imagenet_acc_eval.py ./result ./Imagenet/val_label.txt ./ result.json
      ```
      - 参数说明
        - ./result: 推理结果
        - val_label.txt: 验证label
        - ./: 保存结果路径
        - result.json: 保存结果文件

      结果保存在当前目录的result.json

   4. 性能验证
      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

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
|     310P3     |    1        |  imagenet      |   acc-Top1:73.31<br>acc-Top5:91.44   |  1775.08 |
|     310P3     |    4        |  imagenet      |   acc-Top1:73.31<br>acc-Top5:91.44   |  4113.16 |
|     310P3     |    8        |  imagenet      |   acc-Top1:73.31<br>acc-Top5:91.44   |  5140.21 |
|     310P3     |    16        |  imagenet      |   acc-Top1:73.31<br>acc-Top5:91.44   |  5455.38 |
|     310P3     |    32        |  imagenet      |   acc-Top1:73.31<br>acc-Top5:91.44   |  5527.69 |
|     310P3     |    64        |  imagenet      |   acc-Top1:73.31<br>acc-Top5:91.44   |  5082.96 |