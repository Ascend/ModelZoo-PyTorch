# IntraDA-deeplabv2模型-推理指导


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

IntraDA是无监督域适应模型，该模型使用deeplabv2作为基础语义分割模型，使用带标注的GTA5数据集与无标注的Cityscape数据集训练，并在Cityscape验证集上测试，旨在迁移GTA5中的信息来完成真实场景中的语义分割任务。

- 参考论文：

  [Unsupervised Intra-domain Adaptation for Semantic Segmentation through Self-Supervision](https://arxiv.org/abs/2004.07703)


- 参考实现：

  ```
  url=https://github.com/feipan664/IntraDA
  commit_id=070b0b702fe94a34288eba4ca990410b5aaadc4a
  ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- | -------- | --------------------------- | ------------ |
  | image    | RGB_FP32 | batch_size x 3 x 512 x 1024 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                       | 数据排布格式 |
  | -------- | -------- | -------------------------- | ------------ |
  | output_0 | FP32     | batch_size x 19 x 65 x 129 | NCHW         |
  | output_1 | FP32     | batch_size x 19 x 65 x 129 | NCHW         |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.1   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码。

2. 在同级目录下，获取第三方开源代码仓。

   ```
   git clone https://github.com/feipan664/IntraDA.git
   cd IntraDA
   git reset --hard 070b0b702fe94a34288eba4ca990410b5aaadc4a
   pip3 install -e ./ADVENT
   cd ..
   ```

3. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型使用[Cityscapes数据集](https://www.cityscapes-dataset.com/downloads/)的验证集进行推理测试 ，用户自行获取gtFine_trainvaltest.zip和leftImg8bit_trainvaltest.zip两个压缩包后，将文件解压并上传数据集到指定路径下。数据集目录结构如下所示：

   ```
   cityscapes
   |-- gtFine # 真值标签文件
   |   `-- val
   |       |-- frankfurt
   |       |   |-- frankfurt_000000_000294_gtFine_color.png
   |       |   |-- frankfurt_000000_000294_gtFine_instanceIds.png
   |       |   |-- frankfurt_000000_000294_gtFine_labelIds.png
   |       |   |-- frankfurt_000000_000294_gtFine_labelTrainIds.png
   |       |   |-- frankfurt_000000_000294_gtFine_polygons.json
   |       |   |-- frankfurt_000000_000576_gtFine_color.png
   |       |   |-- ...
   |		...
   |-- leftImg8bit # 原图
   |   `-- val
   |       |-- frankfurt
   |       |   |-- frankfurt_000000_000294_leftImg8bit.png
   |       |   |-- frankfurt_000000_000576_leftImg8bit.png
   |       |   |-- ...
   |		...
   |-- test.txt
   |-- train.txt
   `-- val.txt
   ```
   
2. 数据预处理，将原始数据集转换为模型的输入数据。

   执行intrada_preprocess.py 脚本，完成数据预处理。

   ```
   python3 intrada_preprocess.py ${data_dir} ${save_dir}
   ```
   参数说明：

   - --参数1：原数据集所在路径。
   
   - --参数2：生成数据集二进制文件的所在路径。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从开源仓获取权重文件[cityscapes_easy2hard_intrada_with_norm.pth](https://1drv.ms/u/s!AthTAwNfTh-Yhkt8qVVtIUCT5g4l?e=yKffxB)

   2. 导出onnx文件。

      1. 使用intrada_pth2onnx.py导出**动态**batch的onnx文件。

         ```
         python3 intrada_pth2onnx.py ./cityscapes_easy2hard_intrada_with_norm.pth ./intraDA_deeplabv2.onnx
         ```

         参数说明：

         - --参数1：权重文件。
         - --参数2：生成 onnx 文件。
      
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
         atc --model=intraDA_deeplabv2.onnx \
         --framework=5 \
      --output=intraDA_deeplabv2_bs${bs} \
      --input_format=NCHW \
      --input_shape="image:${bs},3,512,1024" \
      --log=error \
      --soc_version=Ascend${chip_name}
      ```
      
         运行成功后生成intraDA_deeplabv2_bs${bs}.om模型文件。
      
         参数说明：
         - --model：为ONNX模型文件。
         - --framework：5代表ONNX模型。
         - --output：输出的OM模型。
         - --input\_format：输入数据的格式。
         - --input\_shape：输入数据的shape。
         - --log：日志级别。
         - --soc\_version：处理器型号。
      
   
2. 开始推理验证。

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

      ```
      mkdir result
      python3 ${ais_infer_path}/ais_infer.py --model=intraDA_deeplabv2_bs1.om  --batchsize=1 \
      --input ${save_dir} --output result --output_dirname result_bs1
      ```
      
      参数说明：
      
      -   --model：om模型路径。
      -   --batchsize：批次大小。
      -   --input：输入数据所在路径。
      -   --output：推理结果输出路径。
      -   --output_dirname：推理结果输出子文件夹。
   
3. 精度验证。
  
      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据。
   
      ```
    python3 intrada_postprocess.py ${data_dir} ${result_dir}
    ```

      参数说明：
   
      - --参数1：原数据集所在路径。
      - --参数2：推理结果所在路径，例如本文档中应为result/result_bs1。
    
4. 可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
  
      ```
      python3 ${ais_infer_path}/ais_infer.py --model=intraDA_deeplabv2_bs${bs}.om --loop=50 --batchsize=${bs}
      ```
      
      参数说明：
      - --model：om模型路径。
      - --loop：推理循环次数。
      - --batchsize：批次大小。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，IntraDA-deeplabv2模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集     | 精度指标1（MIoU） | 性能（FPS） |
| ----------- | ---------- | ---------- | ----------------- | ----------- |
| Ascend310P3 | 1          | Cityscapes | 47.01%            | 47.52       |
| Ascend310P3 | 4          | Cityscapes | 47.01%            | 43.64       |
| Ascend310P3 | 8          | Cityscapes | 47.01%            | 43.35       |
| Ascend310P3 | 16         | Cityscapes | 47.01%            | 43.41       |
| Ascend310P3 | 32         | Cityscapes | 47.01%            | 42.09       |
| Ascend310P3 | 64         | Cityscapes | 47.01%            | 20.39       |

