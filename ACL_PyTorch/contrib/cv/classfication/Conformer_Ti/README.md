# Conformer_Ti模型-推理指导
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

Conformer_Ti是一种新型的图像分类网络，由卷积神经网络（CNN）和注意力网络（Transformer）两个分类网络组成。另一个主要特征是FCU模块，该模块允许特征信息在两个学习网络之间交互。这些特征允许Conformer_Ti实现更好的分类性能。


- 参考实现：

  ```
  url=https://github.com/pengzhiliang/Conformer
  commit_id=815aaad3ef5dbdfcf1e11368891416c2d7478cb1
  model_name=Conformer_Ti
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

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 23.0.RC1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.3.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.7.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```shell
   git clone https://github.com/pengzhiliang/Conformer.git
   patch -p0 ./Conformer/conformer.py conformer_ti_change.patch
   ```
   
2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```



## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   用户自行获取原始数据集，可选用的开源数据集为ImageNet2012的验证集，将数据集上传到服务器任意路径下并解压。

   ImageNet2012验证集目录结构参考如下所示。

   ```
   ├── ImageNet
      ├── val
      └── val_label.txt 
   ```

2. 数据预处理。

   执行预处理脚本，生成数据集预处理后的bin文件
   ```
   python3 conformer_preprocess.py resnet ImageNet/val ./val_bin
   ```

   第一个参数为数据集类型，该模型为'resnet'，第二个参数为数据集文件位置，第三个为输出bin文件位置及命名



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [Conformer_Ti模型权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Comformer-Ti/PTH/Conformer_tiny_patch16.pth)
   
   2. 导出onnx文件。
   
      1. 使用pth2onnx.py导出onnx文件。
      
         移动pth2onnx.py文件到Conformer源代码文件夹，并运行pth2onnx.py脚本
      
         ```shell
         mv conformer_pth2onnx.py ./Conformer/
         python3 ./Conformer/conformer_pth2onnx.py ./Conformer_tiny_patch16.pth ./conformer_ti.onnx
         ```

         第一个参数为pth文件权重位置，第二个参数为输出onnx文件位置及命名
      
         
      
      2. 优化ONNX文件。
      
         请访问[auto-optimizer推理工具](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)代码仓，根据readme文档进行工具安装。
         
         运行onnx_optimize.py脚本，优化模型。
         ```shell
         python3 onnx_optimize.py --model_path ./conformer_ti.onnx --batch_size ${bs} --save_path ./conformer_ti_bs${bs}.onnx
         ```
         
         - 参数说明：
            - model_path：onnx模型路径。
            - save_path：修改后的onnx模型路径。
            - batch_size：修改后模型batch size。
   
   
   
   3. 使用ATC工具将ONNX模型转OM模型。
   
      1. 配置环境变量。
      
         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
      
         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
      
         
      
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
         atc --framework=5 --model=conformer_ti_bs${bs}.onnx --output=conformer_ti_bs${bs} --input_format=NCHW --input_shape="image:${bs},3,224,224" --log=error --soc_version={chip_name} --op_precision_mode=./op_precision.ini
         ```
      
         - 参数说明：
            - --model：为ONNX模型文件。
            - --framework：5代表ONNX模型。
            - --output：输出的OM模型。
            - --input_format：输入数据的格式。
            - --input_shape：输入数据的shape。
            - --log：日志级别。
            - --soc_version：处理器型号。
            - --op_precision_mode: 高性能模式

         运行成功后生成`conformer_ti_bs${bs}.om`模型文件。


2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  


   2. 执行推理。

      ```shell
      python3 -m ais_bench --model conformer_ti_bs${bs}.om --input val_bin --output out --output_dirname bs${bs} --outfmt TXT 
      ```
      
      - 参数说明：

         - model：om模型路径。
         - input：bin文件路径。
         - output：推理结果保存路径。
         - output_dirname：推理结果子目录。
         - outfmt：输出结果格式。


   3. 精度验证。
      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

      ```shell
      python3 conformer_postprocess.py ./out/bs${bs} ImageNet/val_label.txt ./ result.json
      ```

      第一个参数为ais_bench输出目录，第二个为数据集配套标签，第三个是生成文件的保存目录，第四个是生成的文件名

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```shell
      python3 -m ais_bench --model=conformer_ti_bs${bs}.om --loop=50
      ```

      - 参数说明：

         - model：om模型路径。
         - loop：推理次数。

      `${bs}`表示不同batch的om模型。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度(top1) | 性能 |
| --------- | ---------- | ---------- | ---------- | --------------- |
|Ascend310P3| 1          | ImageNet2012 |   81.09%   | 393.70 |
|Ascend310P3| 4          | ImageNet2012 |   81.09%   | 743.04 |
|Ascend310P3| 8          | ImageNet2012 |   81.09%   | 907.58 |
|Ascend310P3| 16         | ImageNet2012 |   81.09%   | 869.61 |
|Ascend310P3| 32         | ImageNet2012 |   81.09%   | 722.53 |
|Ascend310P3| 64         | ImageNet2012 |   81.09%   | 612.15 |