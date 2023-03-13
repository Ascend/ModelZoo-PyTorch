# VOLO Onnx模型端到端推理指导
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

VOLO采用两阶段架构设计，同时考虑了更具细粒度的token表示编码和全局信息聚合。第一阶段由一堆Outlookers组成，用于生成精细级别的token表示。第二阶段部署一系列transformer blocks来聚合全局信息。VOLO模型在ImageNet-1K分类任务能达到87.1%的准确率，在分割任务上也能达到SOTA级别。

- 参考论文：[VOLO: Vision Outlooker for Visual Recognition](https://arxiv.org/pdf/2106.13112.pdf)

- 参考实现：

  ```
  url=https://github.com/sail-sg/volo
  branch=main
  commit_id=cf35775276a887ccd749ad82c3ba0ac034020cc9
  ```


## 输入输出数据<a name="输入输出数据"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- |---------------------------| ----------------- | -------- |
  | input   | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW       |


- 输出数据

  | 输出数据 | 大小                   | 数据类型 | 数据排布格式 |
  |----------------------| -------- |--------| ------------ |
  | output | batchsize x 1000 | FLOAT32  | ND     |


# 推理环境准备<a name="推理环境准备"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------------ |---------| ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fpies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.10.0  | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 |         |                                                              |



# 快速上手<a name="快速上手"></a>

## 获取源码<a name="获取源码"></a>

1. 获取源码。

   ```
   git clone https://github.com/sail-sg/volo.git
   ```

2. 安装依赖。

   ```
   pip3.7.5 install -r requirements.txt
   ```

3. 安装MagicONNX
   
   ```
   git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
   cd MagicONNX
   pip3.7.5 install .
   cd ..
   ```

​		


## 准备数据集<a name="准备数据集"></a>

1. 获取原始数据集。

   该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，图片与标签分别存放在 /home/datasets/imagenet/val 与 /home/datasets/imagenet/val_label.txt。

   解压后数据集目录结构：
   ```
   imagenet
   ├── val_label.txt    //验证集标注信息       
   └── val             // 验证集文件夹
   ```

2. 数据预处理。

   将原始数据转化为二进制文件（.bin）。

   执行volo_preprocess.py脚本，生成数据集预处理后的bin文件，存放在当前目录下的prep_dataset文件夹中。

   ```
   python3.7.5 volo_preprocess.py --src /home/datasets/imagenet/val --des ./data_bs8 --batchsize 8
   ```
   获得data_bs8路径下的bin文件以及当前目录下的volo_val_bs8.txt文件。

   - 参数说明
     - --src：数据集的路径。
     - --des：生成的bin文件路径。
     - --batchsize: 数据集分成的batchsize。


## 模型推理<a name="模型推理"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从官网获取d1_224_84.2.pth.tar权重文件[d1_224_84.2.pth.tar](https://github.com/sail-sg/volo/releases/download/volo_1/d1_224_84.2.pth.tar)。

   2. 导出onnx文件。

      1. 使用volo_pth2onnx.py导出onnx文件。

         运行使用volo_pth2onnx.py脚本。

         ```
         python3.7.5 volo_pth2onnx.py --src d1_224_84.2.pth.tar --des volo_bs8.onnx --batchsize 8
         ```

         获得volo_bs8.onnx文件。
         - 参数说明：
             - --src: 权重文件路径。
             - --des: 生成的onnx文件。
             - --batchsize: 模型batchsize。

      2. 优化ONNX文件。

         ```
         python3.7.5 modify.py --src volo_bs8.onnx --des volo_modify_bs8.onnx
         ```

         获得volo_modify_bs8.onnx文件。
         - 参数说明：
             - --src: 待修改的onnx文件路径。
             - --des: 生成的onnx文件。

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
      
         使用atc将onnx模型转换为om模型文件，工具使用方法可以参考《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。生成转换batch size为8的om模型的命令如下，对于其他的batch size，可作相应的修改。
         
         ```
         atc --framework=5 --model=volo_modify_bs8.onnx --output=volo_bs8 --input_format=NCHW --input_shape="input:8,3,224,224" --log=debug --soc_version=Ascend${chip_name} --buffer_optimize=off_optimize
         ```
      
         - 参数说明：
           -   --framework：5代表ONNX模型。
           -   --model：为ONNX模型文件。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --buffer_optimize: 是否开启数据缓存优化。off_optimize：表示关闭数据缓存优化。
 
         运行成功后生成volo_bs8.om模型文件。
         
2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3.7.5 -m ais_bench --model volo_bs8.om --batchsize 8 --input ./data_bs8 --output ./result --outfmt "TXT" --device 0
        ```
        - 参数说明：
            - --model: 需要进行推理的om离线模型文件。
            - --batchsize: 模型batchsize。
            - --input: 模型需要的输入，指定输入文件所在的目录即可。
            - --output: 推理结果保存目录。结果会自动创建”日期+时间“的子目录，保存输出结果。可以使用--output_dirname参数，输出结果将保存到子目录output_dirname下。
            - --outfmt: 输出数据的格式。设置为"TXT"用于后续精度验证。
            - --device: 指定NPU运行设备。取值范围为[0,255]，默认值为0。

            推理后的输出默认在当前目录result下。

   3. 精度验证。

        调用volo_postprocess.py脚本，可以获得Accuracy数据，精度结果在屏幕上显示。

        ```
        python3.7.5 volo_postprocess.py --batchsize 8 --result ./result/2023_01_09-03_51_08 --label ./volo_val_bs8.txt
        ```

        - 参数说明：
          - --batchsize: 模型batchsize。
          - --result: 推理结果所在路径（根据具体的推理结果进行修改）。
          - --label: 预处理后生成的标签。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7.5 -m ais_bench --model volo_bs8.om --batchsize 8 --output ./result --loop 1000 --device 0
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

|   芯片型号   | Batch Size |    数据集     | 精度acc1  |   性能    |
|:--------:|:----------:|:----------:|:-------:|:-------:|
|  310P3   |     1      |  ImageNet  | 82.532% | 97.354  |
|  310P3   |     4      |  ImageNet  | 84.154% | 119.757 |
|  310P3   |     8      |  ImageNet  | 84.154% | 124.264 |
|  310P3   |     16     |  ImageNet  | 84.154% | 117.365 |
|  310P3   |     32     |  ImageNet  | 84.161% | 114.969 |
|  310P3   |     64     |  ImageNet  | 84.161% | 108.307 |
