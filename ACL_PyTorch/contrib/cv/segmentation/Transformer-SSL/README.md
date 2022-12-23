# Transformer-SSL模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Transformer-SSL使用不同的IOU阈值，训练多个级联的检测器。它可以用于级联已有的检测器，取得更加精确的目标检测。




- 参考实现：

  ```
  url=https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
  commit_id=c7b20110addde0f74b1fbf812b403d16a59a87a9
  model_name=Transformer-SSL（Backbone:Swin-T,LrSchd:3x）
  ```
  



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | 1 x 3 x 800 x 1216 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | bboxs  | 100 x 5 | FLOAT32  | ND           |
  | labels  | 100 | FLOAT32  | ND           |
  | masks  | 100 x 28 x 28 | FLOAT32  | ND           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.7.1   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
   cd Swin-Transformer-Object-Detection
   git reset --hard c7b20110addde0f74b1fbf812b403d16a59a87a9
   patch -p1 < ../change.patch
   cd ..
   ```

2. 安装依赖
    ```
    pip install -r requirements.txt
    pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.7/index.html
    ```



## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   COCO ： COCO数据集是一个大型的、丰富的物体检测，分割和字幕数据集。这个数据集以scene understanding为目标，主要从复杂的日常场景中截取，图像中的目标通过精确的segmentation进行位置的标定。图像包括91类目标，328,000影像和2,500,000个label。目前为止有语义分割的最大数据集，提供的类别有80 类，有超过33 万张图片，其中20 万张有标注，整个数据集中个体的数目超过150 万个。数据集下载地址：[http://cocodataset.org](http://cocodataset.org)。
   该模型使用coco2017的5千张验证集进行测试，图片与标签分别存放在./data/coco/val2017与./data/coco/annotations/instances_val2017.json。格式如下：
   ```shell
   ├──data 
      └── coco 
          ├──annotations 
             └──instances_val2017.json    //验证集标注信息        
          └── val2017                      // 验证集文件夹
   ```
   

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行Transformer_SSL_preprocess.py脚本，完成预处理。

   ```shell
    python Transformer_SSL_preprocess.py \
           --image_src_path=./data/coco/val2017 \
           --bin_file_path=val2017_bin \
           --input_height=800 \
           --input_width=1216
   ```
   - 参数说明：

     -   --image_src_path：为数据集存放路径。
     -   --bin_file_path：推理后数据存放路径。



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       获取代码仓预训练好的[pth文件](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.pth)

   2. 导出onnx文件。

      1. 运行脚本：

         ```shell
         python Transformer_SSL_pth2onnx.py  
                --config Swin-Transformer-Object-Detection/configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py \
                --checkpoint cascade_mask_rcnn_swin_tiny_patch4_window7.pth \
                --input-img  Swin-Transformer-Object-Detection/tests/data/color.jpg  \
                --output-file model.onnx 
         ```
          - 参数说明：

            -   --config：pytorch模型配置。
            -   --checkpoint：模型权重文件。
            -   --input-img：输入样例图片。
            -   --output-file：保存onnx文件。
         获得model.onnx文件。



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
        atc --model=model.onnx  --framework=5  --output=model_bs1  --input_format=NCHW  --input_shape="image:1,3,800,1216"  --op_precision_mode=op_precision.ini  --log=debug  --soc_version=Ascend${chip_name} 
        ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --op_precision_mode：指定算子高性能模式

           运行成功后生成model_bs1.om模型文件。（该模型只支持bs1）

2. 开始推理验证。

   1. 使用ais_bench工具进行推理。

      ais_bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

    
    2.  执行推理。
        
        ```shell
        python -m ais_bench --model ./model_bs1.om --input ./val2017_bin --output ./ --output_dirname result --batchsize 1
        ```

         - 参数说明：  

           -   --model：om文件路径。
           -   --input：预处理后的bin文件夹路径。
           -   --output:推理结果路径

           >**说明：** 
           >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请[参见](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)。

    
    3.  精度验证。
    
        调用脚本与数据集标签./data/coco/annotations/instances_val2017.json比对，可以获得Accuracy数据。
    
        ```
        python Transformer_SSL_postprocess.py  \
               --ann_file_path=./data/coco/annotations/instances_val2017.json  \
               --bin_file_path=./result  \
               --input_height=800  \
               --input_width=1216 
        ```
    
        - 参数说明：

          -   --ann_file_path：为标签信息文件。
          -   --bin_file_path：为ais_infer推理结果存放路径。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
该模型只支持bs1推理     
| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|    Ascend310P3       |    1 x 3 x 800 x 1216              |   coco2017         |      68.8:66.1      |    4.20             |

