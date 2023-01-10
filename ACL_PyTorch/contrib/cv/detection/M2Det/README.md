# M2Det ONNX模型端到端推理指导
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

特征金字塔被广泛应用于目标检测中(one-stage的DSSD、RetinaNet、RefineDet和two-stage的Mask R-CNN、DetNet)，主要解决物体检测中的目标多尺度问题。M2Det构建了更加高效的特征金字塔，以提高不同尺寸目标的检测准确率。M2Det抽取了主干特征，对特征进行各种融合、操作，提取出金字塔，尺度有大有小在金字塔上进行检测。

- 参考论文：[M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network](https://arxiv.org/pdf/1811.04533.pdf)

- 参考实现：

  ```
  url=https://github.com/qijiezhao/M2Det
  branch=master
  commit_id=de4a6241bf22f7e7f46cb5cb1eb95615fd0a5e12
  ```


## 输入输出数据<a name="输入输出数据"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | ---- |---------------------------| ----------------- | -------- |
  | image    | FP32 | batchsize x 3 x 512 x 512 | NCHW       |


- 输出数据

  | 输出数据 | 大小            | 数据类型 | 数据排布格式 |
  |---------------| -------- |--------| ------------ |
  | boxes    | (batchsize x 32760) x 81 | FLOAT32  | ND     |
  | scores    | batchsize x 32760 x 4 | FLOAT32  | ND     |




# 推理环境准备<a name="推理环境准备"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------------ |---------| ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fpies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.1   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 |         |                                                              |



# 快速上手<a name="快速上手"></a>

## 获取源码<a name="获取源码"></a>

1. 获取源码。

   ```
   git clone https://github.com/VDIGPKU/M2Det.git
   cd M2Det
   git reset --hard de4a6241bf22f7e7f46cb5cb1eb95615fd0a5e12
   patch -p1 < ../M2Det.patch
   sh make.sh
   mkdir weights
   mkdir logs
   mkdir eval
   cd ..
   mkdir result
   ```

2. 安装依赖。

   ```
   pip3.7.5 install -r requirements.txt
   ```

​		


## 准备数据集<a name="准备数据集"></a>

1. 获取原始数据集。
   本模型支持coco2017 5000张图片的验证集。
   用户需自行获取数据集，将instances_val2017.json文件和val2017文件夹解压并上传数据集到服务器任意文件夹。
   coco2014验证集所需文件目录参考（只列出该模型需要的目录）。
    
   数据集目录结构如下:

    ```
       |-- coco2017                // 验证数据集
           |-- instances_val2017.json    //验证集标注信息  
           |-- val2017             // 验证集文件夹
    ```

2. 数据预处理。

   将原始数据转化为二进制文件（.bin）。

   执行M2Det_preprocess.py脚本，生成数据集预处理后的bin文件，存放在当前目录下的pre_dataset文件夹中。

   ```
   python3.7.5 M2Det_preprocess.py --config=configs/m2det512_vgg.py --save_folder=pre_dataset --COCO_imgs=coco_imgs_path --COCO_anns=coco_anns_path
   ```
   
   - 参数说明
       - --config：模型配置文件。
       - --save_folder：预处理后的数据文件保存路径。
       - --COCO_imgs：数据集images存放路径。
       - --COCO_anns：数据集annotations存放路径。
   
   执行gen_dataset_info.py脚本，生成原始图片数据集的info文件，包括了路径和shape。
   
   ```
   python3.7.5 gen_dataset_info.py jpg ${coco_imgs_path}/val2017 coco_images.info
   ```
   
   - 参数说明
       - jpg：输入文件类型，不需要修改。
       - ${coco_imgs_path}/val2017：数据集images存放路径。
       - coco_images.info：预处理后的数据集info文件保存路径。

## 模型推理<a name="模型推理"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从ModelZoo的源码包中获取m2det512_vgg.pth、vgg16_reducedfc.pth权重文件[m2det512_vgg.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/M2Det/PTH/m2det512_vgg.pth)，
   [vgg16_reducedfc.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/M2Det/PTH/vgg16_reducedfc.pth),
   放在目录M2Det/weights下。

   2. 导出onnx文件。

      1. 使用M2Det_pth2onnx.py导出onnx文件。

         运行M2Det_pth2onnx.py脚本。

         ```
         python3.7.5 M2Det_pth2onnx.py --c=M2Det/configs/m2det512_vgg.py --pth=M2Det/weights/m2det512_vgg.pth --onnx=m2det512.onnx
         ```

         获得m2det512.onnx文件。
         - 参数说明：
             - --c：配置文件。
             - --pth：权重文件。
             - --onnx：输出文件名称。

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
      
         使用atc将onnx模型转换为om模型文件，工具使用方法可以参考《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。生成转换batch size为4的om模型的命令如下，对于其他的batch size，可作相应的修改。
         
         ```
         atc --framework=5 --model=m2det512.onnx --input_format=NCHW --input_shape="image:4,3,512,512" --output=m2det512_bs4 --log=debug --soc_version=Ascend${chip_name} --out_nodes="Softmax_1234:0;Reshape_1231:0"
         ```
      
         - 参数说明：
           -   --framework：5代表ONNX模型。
           -   --model：为ONNX模型文件。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --output：输出的OM模型。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --out_nodes: 指定输出节点。指定的输出节点必须放在双引号中，节点中间使用英文分号分隔。node_name必须是模型转换前的网络模型中的节点名称，冒号后的数字表示第几个输出，例如node_name1:0，表示节点名称为node_name1的第1个输出。
               当选择的torch版本不同是可能会改变算子序号，如果torch不同请查看对应onnx文件算子进行相应的修改。
 
         注：若atc执行出错，错误代码为E10016，请使用Netron工具查看对应Reshape节点和Softmax节点，并修改out_nodes参数。

         运行成功后生成m2det512_bs4.om模型文件。
         
2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3.7.5 -m ais_bench --model m2det512_bs4.om --batchsize 4 --input ./pre_dataset --output ./result --outfmt "BIN" --device 0
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

        调用M2Det_postprocess.py脚本，可以获得Accuracy数据，精度结果保存在result/detection-results_0/COCO/detections_val2017_results.json。

        ```
        python3.7.5 M2Det_postprocess.py --bin_data_path=./result/2023_01_08-22_37_53/ --bin_summary_path=./result/2023_01_08-22_37_53_summary.json --test_annotation=coco_images.info --det_results_path==result/detection-results_0_bs4 --net_out_num=2 --prob_thres=0.1 --COCO_imgs=/opt/npu/coco2017/val2017 --COCO_anns=/opt/npu/coco2017/annotations --is_ais_infer
        ```

        - 参数说明：
            - --bin_data_path：推理结果所在路径（根据具体的推理结果进行修改）。
            - --bin_summary_path: 推理结果summary.json文件所在路径（根据具体的推理结果进行修改）。
            - --test_annotation：验证集数据信息。
            - --det_results_path=：生成结果文件。
            - --net_out_num：网络输出类型个数（此处为score,box，2个）。
            - --prob_thres：参数阈值。
            - --COCO_imgs：coco数据集images路径。
            - --COCO_anns：coco数据集annotations路径。
            - --is_ais_infer：使用ais_bench推理工具。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7.5 -m ais_bench --model m2det512_bs4.om --batchsize 4 --output ./result --loop 1000 --device 0
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

|   芯片型号   | Batch Size |    数据集     |  精度   |   性能   |
|:--------:|:----------:|:----------:|:-----:|:------:|
|  310P3   |     1      |  ImageNet  | 37.8% | 45.349 |
|  310P3   |     4      |  ImageNet  | 37.8% | 65.976 |
|  310P3   |     8      |  ImageNet  | 37.8% | 62.795 |
|  310P3   |     16     |  ImageNet  | 37.8% | 62.974 |
|  310P3   |     32     |  ImageNet  | 37.8% | 60.897 |
|  310P3   |     64     |  ImageNet  | 37.8% | 56.011 |
