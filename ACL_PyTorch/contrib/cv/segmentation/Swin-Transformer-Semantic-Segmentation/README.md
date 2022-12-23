# Swin-Transformer-Semantic-Segmentation 模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)
- [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******

  

# 概述

Transformer 在 NLP 领域表现优异，如何将 Transformer 从 NLP 领域应用到 CV 领域？其挑战来自两个领域在尺度与分辨率上差异。NLP 任务中每个词向量的维度是固定的，而 CV 任务中往往图像尺度变化较大；且与文本段落中的单词量相比，图像中的像素分辨率要高得多。为了解决这些问题，作者提出了一种分层 Transformer，通过 Shifted windows(移位窗口) 将自注意力的计算限制在不重叠的局部窗口范围内，同时允许跨窗口连接，从而带来更高的效率。这种分层架构具有在各种尺度上建模的灵活性，且只有相对于图像大小的线性计算复杂度。Swin Transformer 的这些特性使其与广泛的 CV 任务兼容，包括图像分类和密集预测任务，例如目标检测和语义分割。在这些任务上的优异表现表明，Swin Transformer 可以作为 CV 领域的通用主干网络。


- 参考实现：

  ```
  url = https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation
  commit_id = 87e6f90577435c94f3e92c7db1d36edc234d91f6
  model_name = upernet_swin_small_patch4_window7_512x512
  ```
  

## 输入输出数据

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布 |
  | -------- | -------- | ------------------------- | -------- |
  | input    | RGB_FP32 | batchsize x 3 x 512 x 512 | NCHW     |
  
- 输出数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布 |
  | -------- | -------- | --------------------------- | -------- |
  | output   | FLOAT32  | batchsize x 150 x 512 x 512 | ND       |
  
  

------



# 推理环境准备

- 该模型需要以下插件与驱动，版本配套表如下所示：

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.7.1   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手

## 获取源码

1. 获取源码。

   ```shell
   git clone https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation.git   # 克隆仓库的代码
   cd Swin-Transformer-Semantic-Segmentation        								   		 # 切换到模型的代码仓目录
   git reset --hard 87e6f90577435c94f3e92c7db1d36edc234d91f6                     		  # 代码设置到对应的commit_id
   patch -p1<../change.patch															   # 修改源代码
   ```
   
2. 安装依赖。

   ```shell
   pip install -r requirements.txt
   ```

## 准备数据集

1. 获取原始数据集。

   本推理项目使用 ADE20K 的 2000 张验证集图片来验证模型精度，请进入 [ADE20K官网](http://groups.csail.mit.edu/vision/datasets/ADE20K/) 自行下载数据集（需要先注册）。在Swin-Transformer-Semantic-Segmentation 目录中创建data文件夹，ade数据采集存放在data中。最终，验证集原始图片与标注图片分别存放在annotations/validation和images/validation目录下。目录结构如下：

   ```
   ├── data/ade/ADEChallengeData2016/
       ├── annotations/
           ├── validation/
               ├── ADE_val_00000001.png
               ├── ...
               ├── ADE_val_00002000.png
       ├── images/
           ├── validation/
               ├── ADE_val_00000001.jpg
               ├── ...
               ├── ADE_val_00002000.jpg
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   使用mv命令将前处理脚本Swin-Transformer-Semantic-Segmentation_preprocess.py移动至Swin-Transformer-Semantic-Segmentation目录下。然后将目录切换到Swin-Transformer-Semantic-Segmentation。执行前处理Swin-Transformer-Semantic-Segmentation_preprocess.py脚本，完成预处理。

   ```shell
   cd ..
   mv Swin-Transformer-Semantic-Segmentation_preprocess.py Swin-Transformer-Semantic-Segmentation/
   cd Swin-Transformer-Semantic-Segmentation
   
   python Swin-Transformer-Semantic-Segmentation_preprocess.py --config configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k.py --save-dir data/bin/
   ```
   
   参数说明：
   
   + --config: 模型配置文件路径
   + --save-dir: 存放生成的bin文件的目录路径


## 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       获取预训练好的 [pth权重文件](https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_small_patch4_window7_512x512.pth)，下载完成后将权重 pth 文件存放于 Swin-Transformer-Semantic-Segmentation/checkpoint 目录下。

   2. 导出onnx文件。

      1. 使用Swin-Transformer-Semantic-Segmentation_pth2onnx.py 导出onnx文件。

         使用mv命令将Swin-Transformer-Semantic-Segmentation_pth2onnx.py脚本移动至Swin-Transformer-Semantic-Segmentation目录下。执行Swin-Transformer-Semantic-Segmentation_pth2onnx.py。
      
         ```shell
         cd ..
         mv Swin-Transformer-Semantic-Segmentation_pth2onnx.py Swin-Transformer-Semantic-Segmentation/
         cd Swin-Transformer-Semantic-Segmentation
         
         python Swin-Transformer-Semantic-Segmentation_pth2onnx.py --config configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k.py --checkpoint checkpoint/upernet_swin_small_patch4_window7_512x512.pth --onnx swin_bs${bs}.onnx --batchsize ${bs}
         ```
         
         参数说明：
         
         + --config: 模型配置文件路径
         + --checkpoint: 预训练权重所在路径
         + --onnx: 生成ONNX模型的保存路径
         + --batchsize: 模型输入的batchsize，默认为 1
         + --opset-version: ONNX算子集版本，默认为 11
         
         运行结束后，在Swin-Transformer-Semantic-Segmentation目录下会生成.onnx文件。
      
   3. 使用ATC工具将ONNX模型转OM模型。
   
      1. 配置环境变量。
   
         ```shell
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 
         回显如下：
         +--------------------------------------------------------------------------------------------+
         | npu-smi 22.0.0                       Version: 22.0.2                                       |
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 16.6         56                0    / 0              |
         | 0       0         | 0000:5E:00.0    | 0            935  / 21534                            |
         +===================+=================+======================================================+
         ```
      
   3. 执行ATC命令。
      
      ```shell
      atc --framework=5 --model=swin_bs${bs}.onnx --output=swin_bs${bs} --input_format=NCHW --input_shape="input:${bs},3,512,512" --log=null --soc_version=Ascend310${chip_name}
      ```
      
      - 参数说明：
      
         + --model: 为ONNX模型文件。
            + --framework: 5代表ONNX模型。
            + --input_shape: 输入数据的shape。
           + --input_format: 输入数据的排布格式。
           + --output: 输出的OM模型。
           + --log：日志级别。
           + --soc_version: 处理器型号。
           
      
        运行结束后，在Swin-Transformer-Semantic-Segmentation目录下会生成.om文件。
   
2. 开始推理验证。

   1. 使用ais_bench工具进行推理。

      ```shell
      mkdir infer     								#创建存放推理结果的文件夹
      ```
      
   2. 执行推理。
   
        ```shell
        python -m ais_bench --model swin_bs${bs}.om --input data/bin/ --output infer/ --batchsize ${bs}
        ```
   
        参数说明：
   
        + --model: OM模型路径
        + --input: 存放预处理bin文件的目录路径
        + --output: 存放推理结果的目录路径
        
        运行成功后，将会在infer文件夹下生成以年月日和时间作为文件名存放的推理结果。
   
   3. 精度验证。
   
      使用mv命令将Swin-Transformer-Semantic-Segmentation_postprocess.py脚本移动至Swin-Transformer-Semantic-Segmentation目录下。执行该脚本可以获得mIoU精度数据。
   
      ```shell
      cd ..
      mv Swin-Transformer-Semantic-Segmentation_postprocess.py Swin-Transformer-Semantic-Segmentation/
      cd Swin-Transformer-Semantic-Segmentation
      
      python Swin-Transformer-Semantic-Segmentation_postprocess.py --config configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k.py --infer-results infer/${infer_result_datalog}/
      ```
      
      参数说明：
      
      +  --config: 模型配置文件路径
      +  --infer-results: 存放推理结果的目录路径
      
      ${infer_result_datalog}是推理结果存放的文件夹名称。运行成功后，程序会打印出模型的mIoU精度指标。
      
   4. 性能验证。
   
      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
   
        ```shell
        python -m ais_bench --model swin_bs${bs}.om --loop 100 --batchsize ${bs}
        ```
      
      参数说明：
      
      + --model: OM模型路径
      + --input: 存放预处理bin文件的目录路径
      + --loop:推理次数，可选参数，默认为1
      + --batchsize:转换OM模型的batchsize，默认为1

# 模型推理性能&精度

1. 性能对比

   在 310P 设备上，当 batchsize 为 1 时模型的性能为 19.24 fps.

   | 芯片型号    | Batch Size | 数据集 | 精度   | 性能     |
   | ----------- | ---------- | ------ | ------ | -------- |
   | Ascend310P3 | 1          | ADE20K | 48.06% | 19.24fps |
   
   注：当 batchsize 为 4 或更高时，因内存不足导致推理失败，无法获取精度和性能数据。		
