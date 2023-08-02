# MobileNetV1模型-推理指导


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

MobileNetV1是一种基于流水线结构，使用深度级可分离卷积构建的轻量级神经网络，它是将标准卷积拆分为了两个操作：深度卷积（depthwise convolution）和逐点卷积（pointwise convolution），同时提出了两个超参数，分别是宽度乘子和分辨率乘子，用于快速调节模型适配到特定环境。MobileNetV1在尺寸、计算量、速度上的有一定优越性。

- 参考论文：[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

- 参考实现：

  ```
  url=https://github.com/wjc852456/pytorch-mobilenet-v1.git
  branch=master
  commit_id=8b3bde3e525ba6d17b9cabb5feb8ee49a9e1e8e0
  ```

## 输入输出数据<a name="输入输出数据"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 1000 | FLOAT32  | ND           |




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
   git clone https://github.com/wjc852456/pytorch-mobilenet-v1.git
   cd pytorch-mobilenet-v1
   git reset --hard 8b3bde3e525ba6d17b9cabb5feb8ee49a9e1e8e0
   cd ..	
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt	
   ```

​		


## 准备数据集<a name="准备数据集"></a>

1. 获取原始数据集。

   本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。解压后数据集目录结构：

   ```
   └─imagenet
       ├─val               # 评估数据集
       └─val_label.txt 	# 评估数据标签
   ```

2. 数据预处理。

   将原始数据转化为二进制文件（.bin）。

   执行imagenet_torch_preprocess.py脚本，生成数据集预处理后的bin文件，存放在当前目录下的prep_dataset文件夹中。

   ```
   python3 imagenet_torch_preprocess.py resnet /home/HwHiAiUser/dataset/imagenet/val ./prep_dataset	
   ```
   
   - 参数说明
     - 第一个参数指定了图片的预处理方式（不需要修改）。
     - 第二个参数为原始数据验证集所在路径。
     - 第三个参数为输出的二进制文件（.bin）所在路径，每个图像对应生成一个二进制文件。



## 模型推理<a name="模型推理"></a>

- 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从ModelZoo的源码包中获取MobileNet-v1权重文件[mobilenet_sgd_rmsprop_69.526.tar](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/MobileNetV1/PTH/mobilenet_sgd_rmsprop_69.526.tar)。

   2. 导出onnx文件。

      1. 执行脚本。

         ```
         python3 mobilenet-v1_pth2onnx.py mobilenet_sgd_rmsprop_69.526.tar mobilenet-v1.onnx
         ```
         
         获得mobilenet-v1.onnx文件。
         
         - 参数说明：
         
           - 第一个参数为输入的MobileNet_v1权重文件。
           - 第二个参数为输出的ONNX模型文件路径以及名称。
         

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
      
         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
      
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
      
         使用atc将onnx模型转换为om模型文件，工具使用方法可以参考《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。生成转换batch size为32的om模型的命令如下，对于其他的batch size，可作相应的修改。
         
         ```
         atc --framework=5 --model=./mobilenet-v1.onnx --input_format=NCHW --input_shape="image:32,3,224,224" --output=mobilenet-v1_bs32 --log=debug --soc_version=Ascend${chip_name}
         ```
      
         - 参数说明：
         
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
         
           运行成功后生成mobilenet-v1_bs32.om模型文件。
         

- 开始推理验证。

   a.  安装ais_bench推理工具。

     请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   b.  执行推理。

   使用batch size为32的om模型文件进行推理，其他batch size可作相应的修改，推理后的输出在当前目录的result文件夹下。

   ```
   mkdir ./result
   python3 -m ais_bench --model ./mobilenet-v1_bs32.om --input ./prep_dataset --batchsize 32 --output ./result --outfmt "TXT" --device 0
   ```

   - 参数说明：

     -   --model：需要进行推理的om模型。

     -   --input：模型需要的输入，支持bin文件和目录，若不加该参数，会自动生成都为0的数据。

     -   --batchsize：模型batch size 默认为1 。当前推理模块根据模型输入和文件输出自动进行组batch。参数传递的batchszie有且只用于结果吞吐率计算。请务必注意需要传入该值，以获取计算正确的吞吐率。

     -   --output：推理结果输出路径。默认会建立日期+时间的子文件夹保存输出结果。

     -   --outfmt：输出数据的格式,应为"TXT"。

     -    --device：NPU设备编号。



   c.  精度验证。

   调用脚本与数据集标签val\_label.txt比对，生成精度验证结果文件，结果保存在result_bs32.json中。

   ```
   python3 imagenet_acc_eval.py ./result/2022_08_21-23_31_47/ /home/HwHiAiUser/dataset/imagenet/val_label.txt ./ result_bs32.json
   ```

   - 参数说明：
     - ./result/2022_08_21-23_31_47/：为生成推理结果所在路径,请根据ais_bench推理工具自动生成的目录名进行更改。
     - val_label.txt：为标签数据。
     - result_bs32.json：为生成结果文件。

   d. 性能验证。
      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

   ```
   python3 -m ais_bench --model ./mobilenet-v1_bs32.om --batchsize 32 --output ./result --loop 1000 --device 0
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

## 精度对比

|           | Top1 Accuracy (%) | Top5 Accuracy (%) |
| :-------: | :---------------: | :---------------: |
| 310精度  |       69.52       |       89.05       |
| 310P3精度 |       69.52       |       89.05       |
| 310B1精度 | 69.52 | 89.05 |



## 性能对比

| Throughput |   310*4   |   310P3   |   310B1   |
|:---------:|:-------:|:---------:|:-----------:|
| bs1        | 4550.44 | 4076.18  | 1465.73 |
| bs4        |   6736 | 7962.32  | 2207.2 |
| bs8        | 7116.32 | 10373.66 | 2293.9 |
| bs16       |   6938 | 12420.51 | 2266.7 |
| bs32       | 6176.84 | 13990.09 | 2378.8 |
| bs64       | 5899.52 | 6721.18  | 2112.5 |
| **最优batch** | **7116.32** | **13990.09** | **2378.8** |
