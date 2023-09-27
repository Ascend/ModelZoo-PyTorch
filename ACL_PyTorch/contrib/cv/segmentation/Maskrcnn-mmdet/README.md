# Maskrcnn-mmdet模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Maskrcnn是经典的示例分割网络，本模型代码基于mmdetection仓中的maskrcnn修改。



- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection.git
  commit_id=
  code_path=https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn
  model_name=MaskRCNN
  ```
  


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 1216 x 1216 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | ------------------ | ------------ |
  | output1  | FLOAT32  | 100 x 5            | ND           |
  | output2  | INT32    | 100                | ND           |
  | output3  | FLOAT32  | 100 x 80 X 28 X 28 | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.20.alpha  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 7.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.10.1   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone -b v2.8.0 https://github.com/open-mmlab/mmdetection.git
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   cd mmdetection/
   git apply ../mmdet_maskrcnn.patch
   pip3 install -v -e .
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型支持coco2017验证集。用户需自行获取数据集，建立data目录，将coco_val2017数据集放在该目录下。目录结构如下：

   ```
   coco/
   ├── annotations    //验证集标注信息       
   └── val2017        // 验证集文件夹
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行mmdet_preprocess脚本，完成预处理。

   ```
   python3 mmdet_preprocess.py \
      --image_src_path=data/coco/val2017 \
      --bin_file_path=val2017_bin \
      --model_input_height=1216 \
      --model_input_width=1216
   ```
   
   - 参数说明：

           -   --image_src_path：数据原路径。
           -   --bin_file_path：数据保存路径。
           -   --model_input_height：输入图像高。
           -   --model_input_width：输入图像宽。  

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       下载权重文件，链接如下，将权重文件放到mmdetection-2.8.0目录下：
       https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth
       
       命令为
       ```
       wget https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth
       ```

   2. 导出onnx文件。

      1. 导出onnx

         ```
         python3 mmdetection/tools/pytorch2onnx.py \
             mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
             ./mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
             --output-file ./maskrcnn_r50_fpn_1x_bs1.onnx \
             --shape 1216 1216
         ```

      2. 修改onnx模型，将label输出int64改为int32。
         
         命令：

         ```
         python3 label_to_int32.py maskrcnn_r50_fpn_1x_bs1.onnx maskrcnn_r50_fpn_1x_md_bs1.onnx
         ```

         第一个参数为输入onnx，第二个参数为保存的onnx，获得maskrcnn_r50_fpn_1x_md_bs1.onnx文件。


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
         atc \
             --model=maskrcnn_r50_fpn_1x_md_bs1.onnx \
             --framework=5 \
             --output=maskrcnn_r50_fpn_1x \
             --input_format=NCHW \
             --input_shape="input:1,3,1216,1216" \
             --log=error \
             --soc_version=Ascend${chilp_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
         python3 -m ais_bench --model=./maskrcnn_r50_fpn_1x.om --input=./val2017_bin --output=./ --batchsize=1
        ```

        推理后的输出默认在当前目录下。


   3. 精度验证。
      推理结束后执行后处理脚本，会输出bbox map和segm map

      ```
       python3 get_info.py jpg data/coco/val2017/ val2017_jpg.info
       python3.7 mmdet_postprocess.py \
           --bin_data_path=2022_12_20-14_15_48 \
           --test_annotation=val2017_jpg.info \
           --det_results_path=det_result \
           --net_out_num=3 \
           --net_input_height=1216 \
           --net_input_width=1216 \
           --ifShowDetObj \
           --val2017_json_path=./data/coco/annotations/instances_val2017.json

      ```

      - 参数说明：
        - bin_data_path：为生成推理结果所在路径  
        - test_annotation：图片标签信息
        - det_results_path：检测结果路径
        - net_out_num：模型输出结果个数
        - net_input_height：模型输入图像高
        - net_input_width：模型输入图像宽
        - ifShowDetObj：在图片上画出后处理结果
        - val2017_json_path：数据集标签文件信息


   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=./maskrcnn_r50_fpn_1x.om --loop=20 --batchsize=1
        ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 基准精度 | 基准性能 | 精度 | 性能 |
| -------- | ---------- | ------ | ------- | -------- | ---- | ---- |
| 310P3    |      1     |  coco  | bbox map50: 0.588; segm map50: 0.557 | 4.612 fps | bbox map50: 0.59; segm map50: 0.554 | 11.3 fps |
