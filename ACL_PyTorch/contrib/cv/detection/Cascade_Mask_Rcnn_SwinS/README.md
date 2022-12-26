# Cascade-Mask-RCNN-SwinS模型-推理指导

- [Cascade-Mask-RCNN-SwinS模型-推理指导](#cascade-mask-rcnn-swins模型-推理指导)
- [概述](#概述)
  - [输入输出数据](#输入输出数据)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [修改源码](#修改源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能\&精度](#模型推理性能精度)

# 概述

**Cascade-Mask-RCNN-SwinS**

Cascade R-CNN是一种对象检测体系结构，旨在通过增加阈值来解决降解性能的问题。它是R-CNN的多阶段扩展，R-CNN阶段的级联对一个阶段的输出进行了依次训练，以训练下一个阶段。Cascade Mask R-CNN通过将掩码头添加到级联反应，将级联R-CNN扩展到实例分段。


- 参考实现：

  ```shell
  url=https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git
  branch=master
  commit_id=c7b20110addde0f74b1fbf812b403d16a59a87a9
  ```

## 输入输出数据

- 输入数据

  | 输入数据 | 大小                       | 数据类型 |
  | -------- | -------------------------- | -------- |
  | input    | batchsize x 3 x 800 x 1216 | RGB_FP32 |


- 输出数据

  | 输出数据 | 大小          | 数据类型 |
  | -------- | ------------- | -------- |
  | bbox     | 100 x 5       | FLOAT32  |
  | label    | 100           | FLOAT32  |
  | mask     | 100 x 28 x 28 | FLOAT32  |


# 快速上手

## 获取源码

1. 获取源码。

   ```shell
   git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git
   cd Swin-Transformer-Object-Detection
   git checkout master
   git reset --hard c7b20110addde0f74b1fbf812b403d16a59a87a9
   ```

2. 安装依赖。

   ```shell
   pip install -r ../requirements.txt
   ```

## 修改源码

1. 修改源码，完成补丁操作

   ```shell
   patch -p1 < ../swin.patch
   ```

2. 安装源码

   ```shell
   pip install -r requirements/build.txt
   pip install -v -e . 
   ```
## 准备数据集

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用coco2017的5千张验证集进行测试，图片与标签分别存放在./data/coco/val2017与./data/coco/annotations/instances_val2017.json。格式如下：

   ```shell
   ├──data 
      └── coco 
          ├──annotations 
              └──instances_val2017.json    //验证集标注信息        
          └── val2017                      // 验证集文件夹
   ```

2. 数据预处理。

   将原始数据（.jpg）转化为二进制文件（.bin）。以coco_2017数据集为例，通过缩放、均值方差等手段归一化，输出为二进制文件。

   将cascade_maskrcnn_preprocess.py脚本移动到Swin-Transformer-Object-Detection目录下

   Swin-Transformer-Object-Detection目录下执行“cascade_maskrcnn_preprocess.py”脚本，完成预处理。
   
   ```shell
   python3.7 cascade_maskrcnn_preprocess.py \
   --image_src_path=./data/coco/images/val2017 \
   --bin_file_path=val2017_bin \
   --input_height=800 \
   --input_width=1216
   ```

每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“val2017_bin”二进制文件夹。


## 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       使用训练好的pth权重文件：[cascade_mask_rcnn_swin_small_patch4_window7.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_small_patch4_window7.pth)

   2. 导出onnx文件。

      Swin-Transformer-Object-Detection目录下使用tools/deployment/pytorch2onnx.py脚本导出onnx文件。

         运行pytorch2onnx.py脚本。

         ```shell
         python3.7 tools/deployment/pytorch2onnx.py \
         configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py \
         checkpoints/cascade_mask_rcnn_swin_small_patch4_window7.pth \
         --output-file swin-s.onnx
         ```

         获得swin-s.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。
   
         ```shell
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
   
      2. 执行命令查看芯片名称。
   
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
          atc --model=swin-s.onnx \
         --framework=5 \
         --output=swin-s_bs1 \
         --input_format=NCHW \
         --input_shape="input:1,3,800,1216" \
         --log=debug \
         --soc_version=Ascend${芯片类型} \
         --op_precision_mode=op_precision.ini \
         ```
      
         - 参数说明：
      
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --op_precision_mode：设置算子精度模式配置文件（.ini格式）的路径以及文件名。
      

​				运行成功后生成**swin-s_bs1.om**模型文件。

1. 开始推理验证。

   1. 安装ais_bench推理工具。

      请点击本链接进行安装ais_bench推理工具，以及查看具体使用方法(https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)  
   
   2. 执行推理。
   
      ```shell
      python3.7 -m ais_bench --batchsize 1 --model ./swin-s_bs1.om --input "./val2017_bin" --output "result_ais"
      ```
   - 参数说明：
      - model：om文件路径。
      - batchsize：om文件对应的模型batch size。
      - input：模型输入的路径。
      - output：推理结果输出路径。

	
	3. 精度验证。
	将cascade_maskrcnn_postprocess.py脚本移动到Swin-Transformer-Object-Detection目录下
	调用cascade_maskrcnn_postprocess.py评测map精度。
      
         ```shell
         python3.7 cascade_maskrcnn_postprocess.py \
         --ann_file_path=./data/coco/annotations/instances_val2017.json \
         --bin_file_path=./result_ais/ \
         --input_height=800 \
         --input_width=1216 \
         ```

         - 参数说明：

            - --ann_file_path：为原始图片信息文件。
            - --bin_file_path：为ais_bench推理结果。
            - --input_height：输入图片的高
            - --input_height：输入图片的宽   
   
   4. 性能验证
      
      ```shell
      python3.7 -m ais_bench --batchsize 1 --model swin-s_bs1.om --outfmt BIN --loop 20 --output ./performance 
      ```
      - 参数说明：      
         - model：om文件路径。
         - batchsize：om文件对应的模型batch size。
         - input：模型输入的路径。
         - output：推理结果输出路径。


# 模型推理性能&精度
该模型只支持bs1推理
| 模型                         | 官网pth精度                  | 310P推理精度                 | 310P性能 | T4性能 |
| ---------------------------- | ---------------------------- | ---------------------------- | -------- | ------ |
| Cascade-Mask-Rcnn-SwinS(bs1) | bboxmap:51.9<br>maskmap:45.0 | bboxmap:51.4<br>maskmap:44.6 | 3.17      | 2.5    |