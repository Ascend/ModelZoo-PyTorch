# PSPNet模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

参考论文[Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)，使用PPM(pyramid pooling module)和提出的PSPNet(pyramid scene parsing network)，实现了通过融合different-region-based context获取全局context信息的能力。同时，PSPNet在多个数据集上实现了SOTA，取得ImageNet scene parsing challenge 2016、PASCAL VOC 2012 benchmark和Cityscapes benchmark的第1名。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmsegmentation.git
  commit_id=cdd58964b04961fff8abd4f20de1a653c5f6f51c
  code_path=mmsegmentation/mmseg/models/decode_heads/psp_head.py
  model_name=PSPNet
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 500x 500| NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 500 x 500| ND  |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/open-mmlab/mmsegmentation.git
   cd mmsegmentation/
   git reset --hard cdd58964b04961fff8abd4f20de1a653c5f6f51c
   git apply ../PSPNet.patch
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   cd mmsegmentation
   pip3 install -e .
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   该模型使用[VOC2012官网](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)的VOC2012的1449张验证集进行测试，目录结构如下：

   ```
   VOCdevkit/VOC2012/
   ├── JPEGImages          // 测试图片文件夹  
   ├── ImageSets           // 数据集信息文件夹
   └── SegmentationClass   // Ground Truth文件夹
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行pspnet_preprocess.py脚本，完成预处理。

   ```
   python3 pspnet_preprocess.py \
       --image_folder_path=${data_path}/VOCdevkit/VOC2012/JPEGImages/ \
       --split=${data_path}/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt \
       --bin_folder_path=./voc12_bin/
   ```

   ${data_path}表示数据集路径。
   - 参数说明：

     -   --image_folder_path：图片路径。
     -   --split：验证文件。
     -   --bin_folder_path：预处理文件保存路径。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      基于mmsegmentation训练的PSPNet模型的权重文件链接为：

      https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x512_20k_voc12aug/pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pth
      
      获取命令：
      
      ```
      wget https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x512_20k_voc12aug/pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pth
      ```

   2. 导出onnx文件。

      1. 使用pytorch2onnx导出onnx文件。

         运行pytorch2onnx.py脚本。

         ```
         python3 mmsegmentation/tools/pytorch2onnx.py \
             mmsegmentation/configs/pspnet/pspnet_r50-d8_512x512_20k_voc12aug.py \
             --checkpoint pspnet_r50-d8_512x512_20k_voc12aug_20200617_101958-ed5dfbd9.pth \
             --output-file pspnet_bs1.onnx \
             --shape 500 500  
         ```

         命令使用及参数说明可通过`python3 mmsegmentation/tools/pytorch2onnx.py -h`查看。

         获得pspnet_bs1.onnx文件。

      2. 优化ONNX文件。
         
         使用onnx-simplifier工具简化onnx模型，onnx-simplifier工具说明参考[官方链接](https://github.com/daquexian/onnx-simplifier)。

         ```
         onnxsim pspnet_bs1.onnx pspnet_sim_bs${batchsize}.onnx --overwrite-input-shape "input:${batchsize},3,500,500"
         ```

         获得`pspnet_sim_bs${batchsize}.onnx`文件，${batchsize}支持的值为：1，4，8，16，32，64。

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
              --framework=5 \
              --model=pspnet_sim_bs${batchsize}.onnx \
              --output=pspnet_sim_bs${batchsize} \
              --input_format=NCHW \
              --input_shape="input:${batchsize},3,500,500" \
              --log=error \
              --soc_version=Ascend${chip_name} \
              --input_fp16_nodes=input
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --input_fp16_nodes：设置输入为float16的节点。

           运行成功后生成`pspnet_sim_bs${batchsize}.om`模型文件，${batchsize}支持的值为：1，4，8，16，32，64。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
         python3 -m ais_bench \
             --model=./pspnet_sim_bs${batchsize}.om \
             --input=./voc12_bin/ \
             --batchsize=${batchsize} \
             --output=./result \
             --output_dirname=result_bs${batchsize}
        ```
         - 参数说明：
   
           -   --model：om模型路径。
           -   --input：bin文件路径。
           -   --batchsize：om模型的batch。
           -   --output：推理结果保存路径。
           -   --output_dirname：推理结果子文件夹。
           
   
        `${batchsize}`表示不同batch的om模型。推理完成后在指定目录即`./result/result_bs${batchsize}`生成推理结果。

   
   3. 精度验证。
      
      获取图片信息，命令为：
   
      ```
      python3 get_info.py jpg ${data_path}/VOCdevkit/VOC2012/JPEGImages/ voc12_jpg.info
      ```
      
      第一个参数为图片类型，第二个参数为数据集图片路径，第三个参数为生成的图片信息文件。
   
      调pspnet_postprocess.py脚本，计算mIoU指标，命令为：
   
      ```
       python3 pspnet_postprocess.py \
           --test_annotation=./voc12_jpg.info \
           --img_dir=${data_path}/VOCdevkit/VOC2012/JPEGImages \
           --ann_dir=${data_path}/VOCdevkit/VOC2012/SegmentationClass \
           --split=${data_path}/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt \
           --net_input_width=500 \
           --net_input_height=500 \
           --bin_data_path=./result/result_bs${batchsize}
      ```
   
      - 参数说明：
   
        - --test_annotation：图片信息文件。
        - --img_dir：图片路径。
        - --ann_dir：Ground Truth路径。
        - --split：验证文件。
        - --net_input_width：模型输入图片宽。
        - --net_input_height：模型输入图片高。
        - --bin_data_path：om推理结路径。
   
   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3 -m ais_bench --model=pspnet_sim_bs${batchsize}.om --loop=20 --batchsize=${batch_size}
        ```
   
      - 参数说明：
        - --model：om模型。
        - --loop：执行推理次数。
        - --batchsize：om模型的batchsize。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| ----------- | -------- | ------- | ----------- | --------------- |
| Ascend310P3 | 1        | VOC2012 | mIoU: 76.18 | 48.52     |
| Ascend310P3 | 4        | VOC2012 | mIoU: 76.18 | 63.26     |
| Ascend310P3 | 8        | VOC2012 | mIoU: 76.18 | 66.99     |
| Ascend310P3 | 16       | VOC2012 | mIoU: 76.18 | 67.85     |
| Ascend310P3 | 32       | VOC2012 | mIoU: 76.18 | 67.23     |
| Ascend310P3 | 64       | VOC2012 | mIoU: 76.18 | 63.69     |

注意：性能最优batchsize为16。