# RefineDet模型PyTorch离线推理指导
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

RefineDet是发表在2018CVPR上的一篇文章，是SSD算法和RPN网络、FPN算法的结合，可以在保持SSD高效的前提下提高检测效果。一方面引入two stage类型的object detection算法中对box的由粗到细的回归思想，另一方面引入类似FPN网络的特征融合操作用于检测网络，可以有效提高对小目标的检测效果，检测网络的框架还是SSD。

- 参考论文：[Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/pdf/1711.06897.pdf)

- 参考实现：

  ```
  url=https://github.com/luuuyi/RefineDet.PyTorch.git
  branch=master
  commit_id=0e4b24ce07245fcb8c48292326a731729cc5746a
  ```


## 输入输出数据<a name="输入输出数据"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小        | 数据排布格式 |
  | -------- |-----------| ----------------- | -------- |
  | image   | RGB_FP32 | batchsize x 3 x 320 x 320 | NCHW       |


- 输出数据

  | 输出数据         | 大小                   | 数据类型 | 数据排布格式 |
  |----------------------| -------- |--------| ------------ |
  | arm_loc_data | batchsize x 6375 x 4 | FLOAT32  | ND     |
  | arm_conf_data   | batchsize x 6375 x 2 | FLOAT32  | ND     |
  | odm_loc_data   | batchsize x 6375 x 4 | FLOAT32  | ND     |
  | odm_conf_data   | batchsize x 6375 x 21 | FLOAT32  | ND     |




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
   git clone https://github.com/luuuyi/RefineDet.PyTorch.git
   cd RefineDet.PyTorch
   git checkout master 
   git reset --hard   0e4b24ce07245fcb8c48292326a731729cc5746a
   patch -p1 <  ../refinedet.patch
   cd ..
   ```

2. 安装依赖。

   ```
   pip3.7.5 install -r requirements.txt
   ```

​		


## 准备数据集<a name="准备数据集"></a>

1. 获取原始数据集。

   本模型支持VOC2007 4952张图片的验证集。请用户需自行获取[VOC2007数据集](http://host.robots.ox.ac.uk/pascal/VOC)，上传数据集到服务器任意目录并解压（如：/home/datasets/VOCdevkit/）。
   
   解压后数据集目录结构：

   ```
   └─VOCdevkit
       └─VOC2007
           ├──SegmentationObject # 实例分割图像
           ├──SegmentationClass  # 语义分割图像
           ├──JPEGImages         # 训练集和验证集图片
           ├──Annotations        # 图片标注信息（label）
           ├──ImageSets          # 训练集验证集相关数据
           │    ├── Segmentation
           │    ├── Main
           │    └── Layout
   ```

2. 数据预处理。

   将原始数据转化为二进制文件（.bin）。

   执行RefineDet_preprocess.py脚本，生成数据集预处理后的bin文件，存放在当前目录下的voc07test_bin文件夹中。

   ```
   mkdir prepare_dataset
   python3.7.5 RefineDet_preprocess.py '/home/datasets/VOCdevkit/' voc07test_bin
   ```
   
   - 参数说明
     - “/home/datasets/VOCdevkit/”：数据集的路径。
     - “voc07test_bin”：生成的bin文件路径。


## 模型推理<a name="模型推理"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从ModelZoo的源码包中获取RefineDet权重文件[RefineDet320_VOC_final.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/RefineDet/PTH/RefineDet320_VOC_final.pth)。

   2. 导出onnx文件。

      1. 使用RefineDet_pth2onnx.py导出onnx文件。

         运行RefineDet_pth2onnx.py脚本。

         ```
         python3.7.5 RefineDet_pth2onnx.py './RefineDet320_VOC_final.pth'  'RefineDet320_VOC_final_no_nms.onnx' '/home/datasets/VOCdevkit/'
         ```

         获得RefineDet320_VOC_final_no_nms.onnx文件。
         - 参数说明：
             - “./RefineDet320_VOC_final.pth”：权重文件路径。
             - “RefineDet320_VOC_final_no_nms.onnx”：生成的onnx文件。
             - “/home/datasets/VOCdevkit/”：数据集路径。

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
         atc --framework=5 --model=RefineDet320_VOC_final_no_nms.onnx --output=refinedet_voc_320_non_nms_bs16 \
         --input_format=NCHW --input_shape="image:16,3,320,320" --out_nodes="Reshape_239:0;Softmax_246:0;Reshape_226:0;Softmax_233:0"\
         --log=debug --soc_version=Ascend${chip_name} --precision_mode allow_fp32_to_fp16 
         ```
      
         - 参数说明：
           -   --framework：5代表ONNX模型。
           -   --model：为ONNX模型文件。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --out_nodes: 指定输出节点。指定的输出节点必须放在双引号中，节点中间使用英文分号分隔。node_name必须是模型转换前的网络模型中的节点名称，冒号后的数字表示第几个输出，例如node_name1:0，表示节点名称为node_name1的第1个输出。
               当选择的torch版本不同是可能会改变算子序号，如果torch不同请查看对应onnx文件算子进行相应的修改。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --precision_mode: 设置网络模型的精度模式。"allow_fp32_to_fp16"表示如果网络模型中算子支持fp32，则保留原始精度fp32；如果网络模型中算子不支持fp32，则选择fp16。

         注：若atc执行出错，错误代码为E10016，请使用Netron工具查看对应Reshape节点和Softmax节点，并修改out_nodes参数。
         
         运行成功后生成refinedet_voc_320_non_nms_bs16.om模型文件。
         
2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3.7.5 -m ais_bench --model refinedet_voc_320_non_nms_bs16.om --batchsize 16 --input ./voc07test_bin --output ./result --outfmt "BIN" --device 0
        ```
        - 参数说明：
            - --model: 需要进行推理的om离线模型文件。
            - --batchsize: 模型batchsize。
            - --input: 模型需要的输入，指定输入文件所在的目录即可。
            - --output: 推理结果保存目录。结果会自动创建”日期+时间“的子目录，保存输出结果。可以使用--output_dirname参数，输出结果将保存到子目录output_dirname下。
            - --outfmt: 输出数据的格式。设置为"BIN"用于后续精度验证。
            - --device: 指定NPU运行设备。取值范围为[0,255]，默认值为0。

            推理后的输出默认在当前目录result下。

   3. 精度验证。

        调用get_prior_data.py脚本获取get_prior_data数据。结果保存在当前目录下的prior_data.txt文件中。
        ```
        python3.7.5 get_prior_data.py
        ```
        调用RefineDet_postprocess.py脚本，可以获得Accuracy数据，结果保存在result.json中。

        ```
        python3.7.5 RefineDet_postprocess.py --datasets_path '/home/datasets/VOCdevkit/' --result_path result/2023_01_05-02_22_20/
        ```

        - 参数说明：
          - --datasets_path：为数据集目录VOCdevkit文件夹所在路径。
          - --result_path：为生成推理结果所在路径,请根据ais_bench推理工具自动生成的目录名进行更改。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7.5 -m ais_bench --model refinedet_voc_320_non_nms_bs16.om --batchsize 16 --output ./result --loop 1000 --device 0
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

|   芯片型号   | Batch Size |   数据集   |   精度   |   性能    |
|:--------:|:----------:|:-------:|:------:|:-------:|
|  310P3   |     1      | VOC2007 | 79.60% | 379.450 |
|  310P3   |     4      | VOC2007 | 79.58% | 427.239 |
|  310P3   |     8      | VOC2007 | 79.58% | 434.099 |
|  310P3   |     16     | VOC2007 | 79.58% | 445.783 |
|  310P3   |     32     | VOC2007 | 79.58% | 440.789 |
|  310P3   |     64     | VOC2007 | 79.58% | 370.347 |

备注：

- nms放在后处理，在cpu上计算
- onnx转om时，不能使用fp16，否则精度不达标