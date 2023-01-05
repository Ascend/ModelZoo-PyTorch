# SE_ResNet50模型-推理指导


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

为了提高网络的表征能力，SE-ResNet-50在ResNet基础上，更加关注特征通道之间的关系，提出了一种新的架构单元，称为“Squeeze-and-Excitation”（SE）模块。它显式地建模特征通道之间的相互依赖关系，采用一种全新的“特征重标定”策略。具体来说，就是通过学习的方式来自动获取到每个特征通道的重要程度，然后依照这个重要程度去提升有用的特征并抑制对当前任务用处不大的特征。SE模块以最小的额外计算成本为深层架构带来了显著的性能改进。SENets在ILSVRC2017分类比赛上获得了第一名


- 参考实现：

  ```
  model_name=build-in/cv/SE_ResNet50_Pytorch_Infer
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

1. 获取源码

   ```
   git clone {repository_url}        # 克隆仓库的代码
   cd {repository_name}              # 切换到模型的代码仓目录
   git checkout {branch/tag}         # 切换到对应分支
   git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
   cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
   ```

2. 安装依赖

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）


   本模型支持ImageNet验证集。用户需自行获取数据集（或给出明确下载链接），图片与标签分别存放在./dataset/ImageNet/val_union路径与./dataset/ImageNet/val_label.txt文件下。目录结构如下：

   ```
   dataset
   |——ImageNet
      ├── val_union     
      └── val_label.txt             
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行imagenet_torch_preprocess.py脚本，完成预处理。

   ```
   python3 ./imagenet_torch_preprocess.py ./dataset/ImageNet/val_union ./data/ImageNet_bin 
   ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       ```
       wget https://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth
       ```
      执行后在当前目录下获取pth权重文件：se_resnet50-ce0d4300.pth


   2. 导出onnx文件。

      1. 使用SE_ResNet50_pth2onnx.py导出onnx文件

         运行SE_ResNet50_pth2onnx.py脚本。

         ```
         python3 SE_ResNet50_pth2onnx.py ./se_resnet50-ce0d4300.pth ./se_resnet50_dynamic_bs.onnx
         ```

         获得se_resnet50_dynamic_bs.onnx文件。


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
          atc --model=./se_resnet50_dynamic_bs.onnx --framework=5 --input_format=NCHW --input_shape="image:32,3,224,224" --output=./se_resnet50_fp16_bs32 --log=error --soc_version=Ascend${chip_name} --insert_op_conf=./aipp_SE_ResNet50_pth.config --enable_small_channel=1
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --insert\_op\_conf=aipp_SE_ResNet50_pth.config:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据

2. 开始推理验证。

   1. 安装ais_bench推理工具。  

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

         ```
         python3 -m ais_bench --model ./se_resnet50_fp16_bs32.om --batchsize 32 --input ./data/ImageNet_bin --output ./ --output_dirname result --outfmt TXT 
         ```
         - 参数说明
           - --model: om模型
           - --batchsize: 模型batch size
           - --input: 输入数据
           - --output: 输出保存路径
           - --output_dirname: 输出保存文件夹
           - --outfmt: 输出格式


        推理后的输出默认在当前目录result下。

        >**说明：** 
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   3. 精度验证。

      调用vision_metric_ImageNet.py工具脚本与数据集标签val_label.txt比对，可以获得Accuracy Top5数据：

      ```
      python3 ./vision_metric_ImageNet.py ./result/ ./dataset/ImageNet/val_label.txt ./ accuracy_result.json
      ```

      第一个参数为生成推理结果所在路径，第二个参数为标签数据，第三个参数为生成结果文件路径，第四个参数为生成结果文件名称。
      执行后模型精度结果保存在./accuracy_result.json文件中

   4. 性能验证。
      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

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
|  310P3        |   32         |  ImageNet      |  acc@1: 77.36<br>acc@5: 93.76    |   2690   |