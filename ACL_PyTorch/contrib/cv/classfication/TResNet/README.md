# TResNet模型-推理指导


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

TResNet系列是针对ImageNet数据集的图片分类模型。一共有三种型号：TResNet-M，TResNet-L和TResNet-XL，它们的区别仅在深度和通道数量不同。因此，作者在网络的前两个阶段放置了”BasicBlock”层，在后两个阶段放置了“ Bottleneck”层。与ResNet50相比，作者还针对不同的TResNet模型修改了通道数和第三阶段的深度。


- 参考实现：

  ```shell
  url=https://github.com/rwightman/pytorch-image-models.git
  commit_id=5b28ef410062af2a2ab1f27bf02cf33e2ba28ca2
  model_name=TResNet
  ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | output1  | FLOAT32  | batchsize x 1000 | ND           |


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

1. 获取源码

   ```shell
   git clone --branch v0.4.5 https://github.com/rwightman/pytorch-image-models.git
   cd pytorch-image-models
   patch -p1 < ../TResNet.patch
   cd ..                    
   ```
   
2. 安装依赖

   ```shell
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集。目录结构如下：

   ```shell
   ImageNet 
   ├── val_union        
   └── val_label.txt     
   ```
   
2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行TResNet_preprocess.py脚本，完成预处理

   ```shell
   python3 TResNet_preprocess.py ./ImageNet/val_union ./prep_dataset
   ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       ```shell
       wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/TResNet/PTH/model_best.pth.tar
       ```

   2. 导出onnx文件。

      1. 使用TResNet_pth2onnx.py导出onnx文件。

         运行TResNet_pth2onnx.py脚本

         ```shell
         python3 TResNet_pth2onnx.py model_best.pth.tar tresnet_m.onnx
         ```

         获得tresnet_m.onnx文件

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```shell
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

         ```shell
          atc --framework=5 --model=tresnet_m.onnx --output=tresnet_patch16_224_bs1 --input_format=NCHW --input_shape="image:1,3,224,224" --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           
   运行成功后生成`tresnet_patch16_224_bs1.om`模型文件，传入其他batchsize,可以转另外的om

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```shell
          python3 -m ais_bench --model tresnet_patch16_224_bs1.om --input prep_dataset --output ./ --output_dirname result --outfmt TXT
        ```

        -   参数说明：

             -   --model：模型。
             -   --input：模型输入
             -   --output：结果输出路径
             -   --output_dirname: 模型输出结果文件夹
             -   --outfmt：输出结果格式
   
     推理后的输出默认在当前目录result下。

   
3. 精度验证。
   
   调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在acc.json中。
   
      ```shell
       python3.7 TResNet_postprocess.py ./result/ ./ImageNet/val_label.txt ./ acc.json
   ```
   
   - 参数说明：
   
        - result：为生成推理结果所在路径
        - val_label.txt：为标签数据
        - acc.json：为生成结果文件
        

   4. 性能验证

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```shell
         python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型
        - --batchsize：推理张数
        - --loop：循环次数



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size |  数据集  |     精度      | 性能 |
| :------: | :--------: | :------: | :-----------: | :--: |
|  310P3   |     1      | ImageNet | TOP5ACC:94.43 | 1154 |
|  310P3   |     4      | ImageNet | TOP5ACC:94.43 | 2552 |
|  310P3   |     8      | ImageNet | TOP5ACC:94.43 | 3158 |
|  310P3   |     16     | ImageNet | TOP5ACC:94.43 | 3249 |
|  310P3   |     32     | ImageNet | TOP5ACC:94.43 | 2756 |
