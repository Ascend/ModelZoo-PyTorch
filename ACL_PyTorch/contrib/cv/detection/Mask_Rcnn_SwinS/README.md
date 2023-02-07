# Mask-RCNN-SwinS模型-推理指导

- [Mask-RCNN-SwinS模型-推理指导](#mask-rcnn-swins模型-推理指导)
- [概述](#概述)
  - [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [安装依赖](#安装依赖)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能&精度](#模型推理性能精度)

# 概述

**Mask-RCNN-SwinS**

Mask-RCNN是一种对象检测体系结构，其框架基于Faster RCNN的基础上应用FCN，实现物体的高效检测，并同时生成一张高质量的每个个体的分割掩码，能够将物体进行像素级别的分割提取。

- 参考实现：

  ```shell
  url=https://github.com/open-mmlab/mmdetectinon
  branch=master
  commit_id=c14dd6c42efb63f662a63fe403198bac82f47aa6
  ```

## 输入输出数据

- 输入数据

  | 输入数据 | 大小                       | 数据类型 |
  | -------- | -------------------------- | -------- |
  | input    | 1 x 3 x 800 x 1216         | RGB_FP32 |


- 输出数据

  | 输出数据 | 大小          | 数据类型 |
  | -------- | ------------- | -------- |
  | dets     | 100 x 5       | FLOAT32  |
  | labels   | 100           | FLOAT32  |
  | mask     | 100 x 28 x 28 | FLOAT32  |


# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                            | 版本    | 环境准备指导                                                                                          |
| ------------------------------------------------------------    | ------- | ------------------------------------------------------------                                          |
| 固件与驱动                                                      | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                            | 6.0.RC1 | -                                                                                                     |
| Python                                                          | 3.7.5   | -                                                                                                     |
| PyTorch                                                         | 1.8.0   | -                                                                                                     |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |


# 快速上手

## 安装依赖

1. 安装原始依赖

   ```shell
   pip3 install -r requirements.txt
   ```


2. 安装源码依赖

   ```shell
   git clone https://github.com/open-mmlab/mmcv
   cd mmcv && git checkout v1.4.0
   MMCV_WITH_OPS=1 pip3 install -e .
   cd ..
   git clone https://github.com/open-mmlab/mmdetection
   cd mmdetection && git checkout c14dd6c42efb63f662a63fe403198bac82f47aa6
   pip3 install -r requirements/build.txt
   python3 setup.py develop
   patch -p1 < ../swin.patch
   cd ..
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

   执行“data_preprocess.py”脚本，完成预处理。

   ```shell
   python3 data_preprocess.py 
   --image_src_path=./data/coco/val2017 \
   --bin_file_path=val2017_bin \
   --model_input_height=800 \
   --model_input_width=1216
   ```

  每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“val2017_bin”二进制文件夹。
  
  执行`get_info.py`执行得到图片信息文件：
  
  ```
  python3 get_info.py jpg ./data/coco/val2017 coco_jpeg.info
  ```

## 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      使用训练好的pth权重文件：[mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth)

   2. 导出onnx文件。

      使用 `mmdetection/tools/deployment/pytorch2onnx.py` 脚本导出onnx文件。

      运行pytorch2onnx.py脚本。

         ```shell
         python3 mmdetection/tools/deployment/pytorch2onnx.py \
         mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
         mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth \
         --output-file maskrcnn_swin_small_bs1.onnx --batch_size=1
         ```

         获得maskrcnn_swin_small_bs1.onnx文件。

   3. 修改onnx文件

      ```shell
      python3 swin_mod_newroi.py maskrcnn_swin_small_bs1.onnx maskrcnn_swin_small_bs1_fix.onnx
      ```

   4. 使用ATC工具将ONNX模型转OM模型。

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
      
         ```shell
         atc --model=maskrcnn_swin_small_bs1_fix.onnx \
         --framework=5 \
         --output=maskrcnn_swin_small_bs1 \
         --input_format=NCHW \
         --input_shape="input:1,3,800,1216" \
         --log=debug \
         --soc_version=Ascend${chip_name} \
         --optypelist_for_implmode="Gelu" \
         --op_select_implmode=high_performance
         ```
      
         - 参数说明：
      
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --optypelist_for_implmodel：定义其他精度模式的算子列表。
           -   --op_select_implmode：算子性能模式。

​        运行成功后生成**maskrcnn_swin_small_bs1.om**模型文件。

1. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  
   
   2. 执行推理。
   
      ```shell
      python3 -m ais_bench --batchsize 1 --model ./maskrcnn_swin_small_bs1.om --input "./val2017_bin" --output ./ais_results --output_dirname bs1
      ```
   - 参数说明：
      - model：om文件路径。
      - batchsize：om文件对应的模型batch size。
      - input：模型输入的路径。
      - output：推理结果输出路径。
      - output_dirname: 输出文件名。

	
	3. 精度验证。
	调用data_postprocess.py评测map精度。
      
      ```shell
      python3 data_postprocess.py \
      --bin_data_path=./ais_results/bs1 \
      --test_annotation=coco_jpeg.info  \
      --anno_path data/coco/annotations/instances_val2017.json \
      --data_path data/coco/val2017
      ```

      - 参数说明：

        - --bin_file_path：为ais_bench推理结果。
        - --test_annotation: 数据集图片信息文件。
        - --anno_path: 为数据集标签路径。
        - --data_path：为数据集所在路径。

   4. 性能验证
 
      ```shell
      python3 -m ais_bench --batchsize 1 --model ./maskrcnn_swin_small_bs1.om --loop 20 
      ```
      - 参数说明：
         - model：om文件路径。
         - batchsize：om文件对应的模型batch size。
         - input：模型输入的路径。
         - loop：推理循环次数。


# 模型推理性能&精度

该模型只支持bs1推理

| 模型                         | 官网pth精度(IoU=0.50:0.95/area=all/maxDets=100) | 310P推理精度(IoU=0.50:0.95/area=all/maxDets=100) | 310P性能 | T4性能 |
| ---------------------------- | ----------------------------                    | ----------------------------                     | -------- | ------ |
| Mask-Rcnn-SwinS(bs1)         | bboxmap: 48.2                                   | bboxmap: 47.8                                    | 5.17     | 2.5    |
