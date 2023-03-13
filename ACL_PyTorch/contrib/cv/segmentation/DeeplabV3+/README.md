# DeepLabV3+模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

DeepLabV3+就是属于典型的DilatedFCN，它是Google提出的DeepLab系列的第4弹, 它的Encoder的主体是带有空洞卷积的DCNN，可以采用常用的分类网络如ResNet，然后是带有空洞卷积的空间金字塔池化模块（Atrous Spatial Pyramid Pooling, ASPP)），主要是为了引入多尺度信息；相比DeepLabv3，v3+引入了Decoder模块，其将底层特征与高层特征进一步融合，提升分割边界准确度。

   - 参考论文：[Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. (2018)](https://arxiv.org/abs/1802.02611)


   - 参考实现：

      ```
      url=https://github.com/jfzhang95/pytorch-deeplab-xception
      branch=master
      commit_id=9135e104a7a51ea9effa9c6676a2fcffe6a6a2e6
      ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | :--------: | :--------: | :-------------------------: | :------------: |
  | input    | RGB_FP32 | batchsize x 3 x 513 x 513 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | :--------: | :--------: | :-------------------------: | :------------: |
  | output  | FLOAT16  | batchsize x 21 x 513 x 513 | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/jfzhang95/pytorch-deeplab-xception
   cd pytorch-deeplab-xception
   git checkout master
   git reset --hard 9135e104a7a51ea9effa9c6676a2fcffe6a6a2e6
   cd ..
   cp -r pytorch-deeplab-xception/modeling/ ./ 
   ```
   源码目录结构如下：

   ```
   ├──pytorch-deeplab-xception              //开源仓目录
   ├──dataset                               //数据集目录
   ├──modeling                              //转模型依赖的目录
   ├──preprocess_deeplabv3plus_pytorch.py
   ├──post_deeplabv3plus_pytorch.py
   ├──deeplabV3plus_pth2onnx.py
   ├──models.py
   ├──utils.py
   ├──LICENCE
   ├──requirements.txt
   ├──modelzoo_level.txt
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   请用户需自行获取[VOCtrainval_11-May-2012 数据集](https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar)，上传数据集到服务器任意目录并解压（以当前路径"./datasets"为例） VOCtrainval_11-May-2012数据集目录结构如下：

   ```
   ├──datasets
         ├──VOCdevkit
               ├──VOC2012
                     ├── ImageSets
                        └── Segmentation
                           ├── train.txt
                           ├── trainval.txt
                           └── val.txt              //验证集文件列表
                     ├── JPEGImages                 //验证数据集文件夹
                     └── SegmentationsClass         //语义分割集
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行preprocess_deeplabv3plus_pytorch.py脚本，完成预处理。

   ```
   python3 preprocess_deeplabv3plus_pytorch.py ./datasets/VOCdevkit/VOC2012/JPEGImages/ ./prep_bin/ ./datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt
   ```

   - 参数说明：
      - 第一个参数：验证数据集路径
      - 第二个参数：处理后的结果路径
      - 第一个参数：验证集图片列表文件


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [deeplab-resnet.pth.tar](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Deeplabv3%2B/PTH/deeplab-resnet.pth.tar)

   2. 导出onnx文件。

      1. 使用deeplabV3plus_pth2onnx.py导出onnx文件。

         运行deeplabV3plus_pth2onnx.py脚本。

         ```
         python3 deeplabV3plus_pth2onnx.py ./deeplab-resnet.pth.tar ./deeplabv3_plus_res101.onnx
         ```

         获得deeplabv3_plus_res101.onnx文件。

      2. 优化ONNX文件。

         ```
         python3 -m onnxsim deeplabv3_plus_res101.onnx deeplabv3_plus_res101_sim_bs1.onnx --input-shape 1,3,513,513
         ```

         获得deeplabv3_plus_res101_sim_bs1.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）。
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
         atc --model=./deeplabv3_plus_res101_sim_bs1.onnx --framework=5 --output_type=FP16 --output=deeplabv3_plus_res101_sim_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,513,513" --enable_small_channel=1 --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：
            - --model：为ONNX模型文件。
            - --framework：5代表ONNX模型。
            - --output：输出的OM模型。
            - --input\_format：输入数据的格式。
            - --input\_shape：输入数据的shape。
            - --log：日志级别。
            - --soc\_version：处理器型号。
            - --enable\_small\_channel:使能后在channel<=4的卷积层会有性能收益。

           运行成功后生成deeplabv3_plus_res101-sim_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      ```
      mkdir result
      python3 -m ais_bench --model=deeplabv3_plus_res101_sim_bs1.om --input=./prep_bin/ --output=./result/ --output_dirname=bs1 --outfmt=BIN --batchsize=1 --device=0
      ```

      - 参数说明：
         - --model：模型路径。
         - --input：处理后的文件路径。
         - --output：推理结果文件路径。
         - --device：NPU设备编号。
         - --outfmt：输出文件格式。
         - --batchsize：批大小


      推理后的输出默认在文件夹(如：./result/bs1)下。


   3. 精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

      ```
      python3 post_deeplabv3plus_pytorch.py --result_path=./result/bs1/ --label_images=./datasets/VOCdevkit/VOC2012/SegmentationClass/ --labels=./datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt
      ```

      - 参数说明：
        - --result_path：推理结果所在路径。
        - --label_images：标签数据图片文件。
        - --labels：验证集图像名称列表。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度(mIOU) | 性能 |
| :---------: | :----------------: | :----------: | :----------: | :---------------: |
|  Ascend310P  |  1  |VOCtrainval_11-May-2012       |     78.43      |     165.545      |
|  Ascend310P  |  4  |VOCtrainval_11-May-2012       |     78.43      |     162.44      |
|  Ascend310P  |  8  |VOCtrainval_11-May-2012       |     78.43      |     163.559      |
|  Ascend310P  |  16  |VOCtrainval_11-May-2012       |     78.43      |    163.779       |
|  Ascend310P  |  32  |VOCtrainval_11-May-2012       |     78.43      |    83.019       |
|  Ascend310P  |  64  |VOCtrainval_11-May-2012       |     78.43      |    84.96       |