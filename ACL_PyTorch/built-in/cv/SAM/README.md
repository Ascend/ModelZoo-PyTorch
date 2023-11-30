# SAM 推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Segment Anything Model (SAM) 是由 Meta 开源的图像分割大模型，在计算机视觉领域（CV）取得了新的突破。SAM 可在不需要任何标注的情况下，对任何图像中的任何物体进行分割，SAM 的开源引起了业界的广泛反响，被称为计算机视觉领域的 GPT。

- 论文：

  ```
  [Segment Anything](https://arxiv.org/abs/2304.02643)
  Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick
  ```


- 参考实现：

  ```
  url=https://github.com/facebookresearch/segment-anything.git
  commit_id=6fdee8f2727f4506cfbbe553e23b895e27956588
  model_name=sam_vit_b_01ec64
  ```

## 输入输出数据<a name="section540883920406"></a>

SAM 首先会自动分割图像中的所有内容，但是如果你需要分割某一个目标物体，则需要你输入一个目标物体上的坐标，比如一张图片你想让SAM分割Cat或Dog这个目标的提示坐标，SAM会自动在照片中猫或狗进行分割，在离线推理时，会转成encoder模型和decoder模型，其输入输出详情如下：

- encoder输入数据

  | 输入数据 | 数据类型     | 大小               | 数据排布格式 |
  |----------|------------------| ------------------------- | ------------ |
  | x    | FLOAT32 | 1 x 3 x 1024 x 1024 | NCHW         |

- encoder输出数据

  | 输出数据 | 数据类型         | 大小 | 数据排布格式 |
  |-----------| -------- | -------- | ------------ |
  | image_embeddings  | FLOAT32  |  1 x 256 x 64 x 64 | NCHW           |


- decoder输入数据

  | 输入数据 | 数据类型     | 大小               | 数据排布格式 |
  |----------|------------------| ------------------------- | ------------ |
  | image_embeddings    | FLOAT32 | 1 x 256 x 64 x 64 | NCHW         |
  | point_coords    | FLOAT32 | 1 x -1 x 2 | ND         |
  | point_labels    | FLOAT32 | 1 x -1 | ND         |
  | mask_input    | FLOAT32 | 1 x 1 x 256 x 256 | NCHW         |
  | has_mask_input    | FLOAT32 | 1 | ND         |


- decoder输出数据

  | 输出数据 | 数据类型        | 大小 | 数据排布格式 |
  |-----------| -------- | -------- | ------------ |
  | iou_predictions  | FLOAT32  | -1 x 1  | ND           |
  | low_res_masks  | FLOAT32  |  -1 x 1 x -1 x -1 | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------------ |---------| ------------------------------------------------------------ |
| 固件与驱动                                                   | 23.0.rc3.b070  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 7.0.T10 | -                                                            |
| Python                                                       | 3.8.13   | -                                                            |
| PyTorch                                                      | 1.13.1   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/facebookresearch/segment-anything.git
   cd segment-anything
   git reset --hard 6fdee8f2727f4506cfbbe553e23b895e27956588
   pip3 install -e .
   cd ..
   patch -p1 < segment_anything_diff.patch
   ```

2.  安装依赖。

    1. 安装基础环境
    ```bash
    pip3 install -r requirements.txt
    ```
    说明：某些库如果通过此方式安装失败，可使用pip单独进行安装。

    2. 安装量化工具

        参考[AMCT(ONNX)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/70RC1alpha001/developmenttools/devtool/atlasamctonnx_16_0004.html)主页安装量化工具。

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   测试[数据图片下载地址](https://segment-anything.com/demo)

   说明：下载图片数据后存放在segment-anything目录下自己新建的data文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       下载对应的[sam_vit_b_01ec64.pth权重文件](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)于segment-anything目录下models中，没有models就自己新建。

   2. 导出onnx文件。

      1. 使用export_onnx_model.py导出onnx文件。

         运行export_onnx_model.py脚本。

         ```
         cd segment-anything
         python3 scripts/export_onnx_model.py  \
             --checkpoint models/sam_vit_b_01ec64.pth  \
             --model-type vit_b  \
             --opset 14 \
             --encoder-output models/encoder.onnx \
             --decoder-output models/decoder.onnx \
             --return-single-mask

         ```

         - 参数说明
         - --checkpoint ：模型权重文件路径。
         - --model-type ：模型类型。
         - --opset ：onnx算子集版本。
         - --encoder-output ：保存encoder模型的输出ONNX模型的文件路径。
         - --decoder-output ：保存decoder模型的输出ONNX模型的文件路径。
         - --return-single-mask : 设置最优mask模式。

            > **说明：**
            生成的encoder.onnx和decoder.onnx文件位于models目录下.

      2. 利用onnxsim工具优化onnx模型。

         ```
         onnxsim models/encoder.onnx models/encoder_sim.onnx
         onnxsim models/decoder.onnx models/decoder_sim.onnx

         ```
          > **说明：**
         第一个参数指onnx原模型，第二个指onnxsim优化之后的模型.
      
      3. 运行decoder_onnx_modify.py脚本修改decoder.onnx模型。

         ```
         cd ..
         python3 decoder_onnx_modify.py ./segment-anything/models/decoder_sim.onnx ./segment-anything/models/decoder_modify.onnx

         ```

         > **说明：**
         第一个参数指decoder.onnx原模型，第二个指修改之后的模型.

   3. 模型量化

      a、encoder模型量化

         在量化前，先生成校验数据，以确保量化后模型精度不会损失：

         ```
         export PYTHONPATH="${PYTHONPATH}:`pwd`/segment-anything"
         python3 sam_quant_preprocessing.py \
               --src-path ./segment-anything/data/demo.jpg \
               --encoder-quant-save-path ./segment-anything/encoder_quant_bin \
               --encoder-quant

         ```
         - 参数说明
         - --src_path: 数据地址
         - --encoder-quant-save-path: 生成校验数据bin文件地址
         - --encoder-quant: 生成encoder量化的校验数据

         然后使用`amct`工具，对ONNX模型进行量化，以进一步提升模型性能：

         ```
         amct_onnx calibration \
               --model=./segment-anything/models/encoder_sim.onnx \
               --save_path=./segment-anything/models/encoder_quant \
               --input_shape="x:1,3,1024,1024" \
               --data_dir=./segment-anything/encoder_quant_bin/x \
               --data_types="float32"

         ```

         - 参数说明
         - --model: onnx模型
         - --save_path: 保存量化后onnx模型文件地址
         - --input_shape: 模型输入shape
         - --data_dir: 校验数据
         - --data_types: 数据类型
      
         量化后的模型存放路径为 `models/encoder_quant_deploy_model.onnx`。

      b、decoder模型量化

         在量化前，先生成校验数据，以确保量化后模型精度不会损失：

         ```
         python3 sam_quant_preprocessing.py \
               --src-path ./segment-anything/data/demo.jpg \
               --encoder-onnx-model-path ./segment-anything/models/encoder_sim.onnx \
               --decoder-quant-save-path ./segment-anything/decoder_quant_bin \
               --input-point "[[500, 375], [1125, 625], [1520, 625]]" \
               --decoder-quant
         ```

         - 参数说明
         - --src_path: 测试数图片地址（自己根据实际图片路径进行更改）
         - --decoder-quant-save-path: 生成校验数据bin文件地址
         - --encoder-onnx-model-path: encoder模型路径
         - --input-point ：输入坐标，根据实际待分割目标物体上指定的点进行修改
         - --decoder-quant: 生成decoder量化的校验数据

         然后使用`amct`工具，对ONNX模型进行量化，以进一步提升模型性能：

         ```
         decoder_q=./segment-anything/decoder_quant_bin
         amct_onnx calibration \
               --model=./segment-anything/models/decoder_modify.onnx \
               --save_path=./segment-anything/models/decoder_quant \
               --input_shape="image_embeddings:1,256,64,64;point_coords:1,-1,2;point_labels:1,-1;mask_input:1,1,256,256;has_mask_input:1"  \
               --data_dir="${decoder_q}/image_embedding;${decoder_q}/point_coord;${decoder_q}/point_label;${decoder_q}/mask_input;${decoder_q}/has_mask_input" \
               --data_types="float32;float32;float32;float32;float32" 
         ```

         - 参数说明
            - --model: onnx模型
            - --save_path: 保存量化后onnx模型文件地址
            - --input_shape: 模型输入shape
            - --data_dir: 校验数据
            - --data_types: 数据类型
      
         量化后的模型存放路径为 `models/decoder_quant_deploy_model.onnx`。
   
   4. 使用ATC工具将ONNX模型转OM模型。

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

         a、encoder模型转om
            ```
            atc --framework=5 \
               --model=segment-anything/models/encoder_quant_deploy_model.onnx \
               --output=segment-anything/models/encoder_quant \
               --input_format=NCHW \
               --input_shape="x:1,3,1024,1024" \
               --op_select_implmode=high_performance \
               --soc_version=Ascend${chip_name} \
               --log=error
            ```

            - 参数说明：
               -  --model：为ONNX模型文件。
               - --framework：5代表ONNX模型。
               - --output：输出的OM模型。
               - --input_format：输入数据的格式。
               - --input_shape：输入数据的shape。
               - --log：日志级别。
               - --soc_version：处理器型号。
               - --input_format：输入数据的格式。
               - --op_select_implmode:高性能模式。
         
            运行成功后在models目录下生成**encoder_quant.om**模型文件。

         b、decoder模型转om
            ```
            atc --framework=5 \
               --model=segment-anything/models/decoder_quant_deploy_model.onnx \
               --output=segment-anything/models/decoder_quant \
               --input_format=ND \
               --input_shape="image_embeddings:1,256,64,64;point_coords:1,-1,2;point_labels:1,-1;mask_input:1,1,256,256;has_mask_input:1" \
               --dynamic_dims="2,2;3,3;4,4;5,5;6,6;7,7;8,8;9,9" \
               --op_select_implmode=high_performance \
               --soc_version=Ascend${chip_name} \
               --log=error
            ```

            - 参数说明：
               -  --model：为ONNX模型文件。
               - --framework：5代表ONNX模型。
               - --output：输出的OM模型。
               - --input_format：输入数据的格式。
               - --input_shape：输入数据的shape。
               - --dynamic_dims:decoder输入分档参数设置
               - --log：日志级别。
               - --soc_version：处理器型号。
               - --input_format：输入数据的格式。
               - --op_select_implmode:高性能模式。
         
            运行成功后在models目录下生成**decoder_quant.om**模型文件。
            > **说明：**
            --dynamic_dims分档设置参数具体分为多少档根据实际情况可自行设置。



2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。


   b.  执行推理。
      使用 sam_end2end_infer.py进行SAM模型离线端到端推理。

      ```
      python3 sam_end2end_infer.py \ 
            --src-path ./segment-anything/data/demo.jpg \
            --save-path ./segment-anything/outputs \
            --encoder-model-path ./segment-anything/models/encoder_quant.om \
            --decoder-model-path ./segment-anything/models/decoder_quant.om \
            --input-point "[[500, 375], [1125, 625], [1520, 625]]" \
            --device-id 0
    
      ```
    
      -   参数说明：
    
           -   --src-path：图片数据路径。
           -   --save-path：SAM离线推理保存路径。
           -   --encoder-model-path：encoder模型路径。
           -   --decoder-model-path：decoder模型路径。
           -   --input-point：分割目标上的坐标（坐标数量根据带分割目标物体分割效果确定，不同的图片数据，不同的待分割目标坐标不一样）
           -   --device-id：NPU卡ID

   

   c. 推理结果图片展示。

      在线模型推理结果：
      ![](./assets/pth_truck_result.JPG)
    
      离线模型推理结果：
      ![](./assets/om_truck_result.JPG)



   d.  性能验证。

      1. encoder模型性能验证
      
      可使用ais_bench推理工具的纯推理模式验证om模型的性能，参考命令如下：
    
      ```
      python3 -m ais_bench --model ./segment-anything/models/encoder_quant.om --loop 100 --batchsize 1
    
      ```
      - 参数说明
        - --model: om模型
        - --loop: 循环次数
        - --batchsize: 模型batch size
    
      2. decoder模型性能验证
      
      可使用ais_bench推理工具的纯推理模式验证om模型的性能，以输入3个坐标为例，参考命令如下：
    
      ```
      python3 -m ais_bench --model ./segment-anything/models/decoder_quant.om --dymDims "image_embeddings:1,256,64,64;point_coords:1,4,2;point_labels:1,4;mask_input:1,1,256,256;has_mask_input:1" --outputSize 1000,1000000 --loop 100 --batchsize 1
    
      ```
      - 参数说明
        - --model: om模型
        - --outputSize:动态模型输出Size设置
        - --auto_set_dymdims_mode：开启动态dims模式
        - --loop: 循环次数
        - --batchsize: 模型batch size


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号  | 模型| Batch Size | 性能    |
|----------|-------|-------|-------|
|  310P3  | encoder   |   1       |  4fps |
|  310P3  | decoder   |   1       | 464fps |
