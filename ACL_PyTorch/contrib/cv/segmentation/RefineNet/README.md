# RefineNet Onnx模型端到端推理指导
- [概述](#概述)
    - [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)

- [快速上手](#快速上手)

  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)

- [模型推理性能](#模型推理性能)

  ******

  

# 概述<a name="概述"></a>

RefineNet是发表在2017CVPR上的一篇文章，旨在实现高分辨的语义分割任务。目前在PASCAL VOC 2012数据集上取得了最好的效果。从网络结构来看，本工作是U-Net的一个变种。文章的主要贡献和创新在于U-Net折返向上的通路之中。

- 参考论文：[RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation](https://arxiv.org/pdf/1611.06612.pdf)

- 参考实现：

  ```
  url=https://github.com/DrSleep/refinenet-pytorch.git
  branch=master
  commit_id=8f25c076016e61a835551493aae303e81cf36c53
  ```
[RefineNet(in Pytorch)](https://github.com/DrSleep/refinenet-pytorch)这个仓库的代码只给出了模型代码，没有给出训练代码，因此RefineNet的训练流程使用了该作者的另一个仓库[light-weight-refinenet](https://github.com/DrSleep/light-weight-refinenet)的训练代码搭配RefineNet的模型代码。
- 参考实现：  
  ```
  url=https://github.com/DrSleep/light-weight-refinenet.git
  branch=master
  commit_id=538fe8b39327d8343763b859daf7b9d03a05396e
  ```


## 输入输出数据<a name="输入输出数据"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- |---------------------------| ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 500 x 500 | NCHW         |


- 输出数据

  | 输出数据 | 大小                         | 数据类型 | 数据排布格式 |
  |----------------------------| -------- |--------| ------------ |
  | output1  | batchsize x 21 x 125 x 125 | FLOAT32  | NCHW   |




# 推理环境准备<a name="推理环境准备"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------------ |---------| ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fpies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.5.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 |         |                                                              |



# 快速上手<a name="快速上手"></a>

## 获取源码<a name="获取源码"></a>

1. 获取源码。

   ```
   git clone https://github.com/DrSleep/refinenet-pytorch.git RefineNet_pytorch
   cd RefineNet_pytorch
   git apply ../RefineNet.patch
   cd ..
   ```

2. 安装依赖。

   ```
   pip3.7.5 install -r requirements.txt	
   git clone https://github.com/drsleep/densetorch.git
   cd densetorch
   pip3.7.5 install -e .
   cd ..
   ```

​		


## 准备数据集<a name="准备数据集"></a>

1. 获取原始数据集。

   模型使用SBD的5623张训练图片以及VOC2012的1464张训练图片作为训练集，VOC2012的1449张验证图片作为验证集，推理部分只需要用到这1449张验证图片。下载VOC2012数据集后，把VOCdevkit文件夹放在/opt/npu下。 
   
   解压后数据集目录结构：

   ```
   └─VOCdevkit
       └─VOC2012
           ├─Annotations  # 图片标注信息（label）
           ├─ImageSets    # 训练集验证集相关数据
           ├─JPEGImages   # 训练集和验证集图片
           ├─SegmentationClass # 语义分割图像
           └─SegmentationObject # 实例分割图像
       
   ```

2. 数据预处理。

   将原始数据转化为二进制文件（.bin）。

   执行RefineNet_preprocess.py脚本，生成数据集预处理后的bin文件，存放在当前目录下的prepare_dataset文件夹中。

   ```
   mkdir prepare_dataset
   python3.7.5 RefineNet_preprocess.py --root-dir /opt/npu/VOCdevkit/VOC2012 --bin-dir ./prepare_dataset
   ```
   
   - 参数说明
     - --root-dir: 数据集所在路径。
     - --bin-dir: 输出的二进制文件（.bin）所在路径，每个图像对应生成一个二进制文件。


## 模型推理<a name="模型推理"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从ModelZoo的源码包中获取[RefineNet权重文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/RefineNet/PTH/RefineNet_910.pth.tar)，重命名为 RefineNet.pth.tar, 放置在当前目录下。

   2. 导出onnx文件。

      1. 使用RefineNet_pth2onnx.py导出onnx文件。

         运行RefineNet_pth2onnx.py脚本。

         ```
         python3.7.5 RefineNet_pth2onnx.py --input-file RefineNet.pth.tar --output-file RefineNet.onnx
         ```

         获得RefineNet.onnx文件。
         - 参数说明：
         
           - --input-file：输入的RefineNet模型的权重文件。
           - --output-file：输出的ONNX模型文件路径以及名称。

   3. 使用ATC工具将ONNX模型转OM模型

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
         
         > **说明：** 
         该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
      
      2. 执行命令查看芯片名称（$\{chip\_name\}）。
      
         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         回显如下：
         +-------------------|-----------------|------------------------------------------------------+
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
      
         使用atc将onnx模型转换为om模型文件，工具使用方法可以参考《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。生成转换batch size为16的om模型的命令如下，对于其他的batch size，可作相应的修改。
         
         ```
         atc --framework=5 --model=RefineNet.onnx --output=RefineNet_bs1 --input_format=NCHW --input_shape="input:1,3,500,500" --log=debug --soc_version=Ascend${chip_name}
         ```
      
         - 参数说明：

           -   --framework：5代表ONNX模型。
           -   --model：为ONNX模型文件。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
         
           运行成功后生成RefineNet_bs1.om模型文件。
         
2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        mkdir result
        python3.7.5 -m ais_bench --model RefineNet_bs1.om --input ./prepare_dataset --batchsize 1 --output ./result --outfmt "TXT" --device 0
        ```
        - 参数说明：

          - --model：需要进行推理的om模型。

          - --input：模型需要的输入，支持bin文件和目录，若不加该参数，会自动生成都为0的数据。

          - --batchsize：模型batchsize。不输入该值将自动推导。当前推理模块根据模型输入和文件输出自动进行组batch。参数传递的batchszie有且只用于结果吞吐率计算。请务必注意需要传入该值，以获取计算正确的吞吐率。

          - --output：推理结果输出路径。默认会建立"日期+时间"的子文件夹保存输出结果。

          - --outfmt：输出数据的格式,本模型应为"TXT"，用于后续精度验证。

          - --device：指定NPU运行设备。取值范围为[0,255]，默认值为0。

        推理后的输出默认在当前目录result下。

   3. 精度验证。

        调用RefineNet_postprocess.py脚本推理结果与语义分割真值进行比对，可以获得IoU精度数据。结果保存在result.json中。

        ```
        ulimit -n 10240
        python3.7.5 RefineNet_postprocess.py --val-dir /opt/npu --result-dir ./result/2023_01_04-21_46_09
        ```

        - 参数说明：
          - --val-dir：为数据集目录VOCdevkit文件夹所在路径。
          - --result-dir：为生成推理结果所在路径,请根据ais_bench推理工具自动生成的目录名进行更改。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7.5 -m ais_bench --model RefineNet_bs1.om --batchsize 1 --output ./result --loop 1000 --device 0
        ```

      - 参数说明：
        - --model：需要进行推理的om模型。
        - --batchsize：模型batchsize。不输入该值将自动推导。当前推理模块根据模型输入和文件输出自动进行组batch。参数传递的batchszie有且只用于结果吞吐率计算。请务必注意需要传入该值，以获取计算正确的吞吐率。
        - --output: 推理结果输出路径。默认会建立"日期+时间"的子文件夹保存输出结果。
        - --loop: 推理次数。默认值为1，取值范围为大于0的正整数。
        - --device: 指定NPU运行设备。取值范围为[0,255]，默认值为0。

   ​	

# 模型推理性能&精度<a name="模型推理性能&精度"></a>

调用ACL接口推理计算，精度和性能参考下列数据。

|   芯片型号   | Batch Size |   数据集   |  精度miou  |   性能   |
|:--------:|:----------:|:-------:|:--------:|:------:|
|  310P3   |     1      | VOC2012 | 0.786359 | 87.713 |
|  310P3   |     4      | VOC2012 | 0.786359 | 83.485 |
|  310P3   |     8      | VOC2012 | 0.786359 | 74.999 |
|  310P3   |     16     | VOC2012 | 0.786359 | 69.635 |
|  310P3   |     32     | VOC2012 | 0.786359 | 61.252 |
|  310P3   |     64     | VOC2012 | 0.786359 | 57.694 |
