 Faster_R-CNN_DCN_Res50模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>


FasterRCNN-DCN是FasterRCNN与DCN可行变卷积相结合得到的网络模型。其相关信息可参考mmdetection仓库。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn
  branch=master
  commit_id=a21eb25535f31634cef332b09fc27d28956fb24b
  model_name=Faster_R-CNN_DCN_Res50
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | 1 x 3 x 1216 x 1216       | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | boxes    | 100 × 5 | FLOAT32  | ND           |
  | labels   | 100 × 1 | INT64    | ND           |



# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.17 | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

1. 下载本模型代码包，并上传至服务器解压至用户目录下


2. 安装常规依赖。

   ```
   pip3.7 install -r requirment.txt
   ```
3. 安装mmcv。(注：此步骤安装时间较长，约10分钟左右，请耐心等候)
   ```
   git clone https://github.com/open-mmlab/mmcv -b master
   cd mmcv
   git checkout v1.2.7
   MMCV_WITH_OPS=1 pip3.7 install -e .
   patch -p1 < ../mmcv.patch
   cd ..
   ```
4. 安装mmdetection。(注：此步骤安装时间较长，约5分钟左右，请耐心等候)

   ```
   git clone https://github.com/open-mmlab/mmdetection.git -b master
   cd mmdetection
   git reset --hard a21eb25535f31634cef332b09fc27d28956fb24b
   patch -p1 < ../dcn.patch
   pip3.7 install -r requirements/build.txt
   python3.7 setup.py develop
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型推理数据集采用 [coco_val_2017](https://cocodataset.org/#download) ，请用户自行获取coco_val_2017数据集。将instances_val2017.json文件和val2017文件夹上传并解压数据集到 `./data/coco` 目录下（如没有则需要创建）。其目录结构如下所示:

   ```
    |--data/coco
        |--instances_val2017.json   //验证集标注信息
        |--val2017                  //验证集文件夹
        |--其他文件以及文件夹
   ```

2. 数据预处理。

   2.1：执行FasterRCNN+FPN+DCN_preprocess.py，将数据集转换(.jpg)为二进制数据(.bin)文件。

   ```
   python3.7 FasterRCNN+FPN+DCN_preprocess.py --image_folder_path ./data/coco/val2017 --bin_folder_path coco2017_bin
   ```

   - 参数说明：
        
        - --image_folder_path：原始数据验证集（.jpg）所在路径。

        - --bin_folder_path：输出的二进制文件（.bin）所在路径。
    
    成功运行后生成val2017_bin文件夹
    
    2.2：执行gen_dataset_info.py，以val2017文件夹的jpg图片生成coco2017_jpg.info文件
   ```
   python3.7 gen_dataset_info.py jpg ./data/coco/val2017 coco2017_jpg.info
   ```
     - 参数说明：

       -   参数1(jpg)：输入数据的数据格式
        
       -   参数2(./val2017)：输入数据的文件夹路径
        
       -   参数3(coco2017_jpg.info)：生成的info信息文件
    
       成功运行后生成coco2017_jpg.info文件


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。(TODO)
        点击[此链接](https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco_20200203-4f85c69c.pth)
        下载经过训练的faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco模型权重文件，并移动到Modelzoo源码包中。

   2. pth导出onnx文件。

      1. 生成onnx模型。

         调用mmdete/tools目录中的pytorch2onnx脚本生成onnx模型。这里注意指定shape为1216。当前框架限制，仅支持batchsize=1的场景。

         ```
         python3.7 mmdetection/tools/pytorch2onnx.py mmdetection/configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py mmdetection/checkpoints/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth   --output-file faster_rcnn_r50_fpn_1x_coco.onnx  --shape 1216 --show
         ```

         获得faster_rcnn_r50_fpn_1x_coco.onnx文件。

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
         +-------------------|-----------------|------------------------------------------------------+
         | npu-smi 22.0.0                       Version:22.0.2                                        |
         +-------------------|-----------------|------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 16.2         42                0    / 0              |
         | 0       0         | 0000:86:00.0    | 0            994  / 21534                            |
         +===================+=================+======================================================+
         ```

      3. 执行ATC命令。

         ```
         atc --framework=5 --model=./faster_rcnn_r50_fpn_1x_coco.onnx --output=./faster_rcnn_r50_fpn_1x_coco_bs1  --input_format=NCHW --input_shape="input:1,3,1216,1216" --log=debug --soc_version=Ascend${chip_name}
         ```

        - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成faster_rcnn_r50_fpn_1x_coco_bs1.om模型文件。



2. 开始推理验证。

   a.  使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]


   b.  执行推理。

      ```
      python3.7 -m ais_bencn --model ./faster_rcnn_r50_fpn_1x_coco_bs1.om --input ./coco2017_bin --output ./ais_results --outfmt BIN --batchsize 1 --output_dirname bs1
      ```
      -   参数说明：
        -   --model：om文件路径。
        -   --input：输入文件。
        -   --output：输出文件所存目录。
        -   --output_dirname: 输出文件名。
        -   --outfmt: 推理结果保存格式。
        -   --batchsize： 模型对应batchsize。

      推理后的输出默认在当前目录ais_result/bs1下。

c.  精度验证。
-   bin转txt：

    ```
    python3.7 FasterRCNN+FPN+DCN_postprocess.py --test_annotation coco2017_jpg.info --bin_data_path ais_result/bs1
    ```
    -   参数说明：
        --bin_data_path：推理结果所在文件夹。
        --test_annotation: 预处理得到info文件。

    生成detection-results文件夹
-   txt转json：
    ```
    python3.7 txt2json.py --npu_txt_path detection-results --json_output_file coco_detection_result
    ```
    -   参数说明：
        --npu_txt_path：后处理得到的detection结果。
        --json_output_file: 转换得到的json文件名。

    生成coco_detection_result.json文件
-   json对比获取精度数据：
    ```
    python3.7 coco_eval.py --ground_truth data/coco/annotations/instances_val2017.json --detection_result coco_detection_result.json
    ```
    -   参数说明：
        --ground_truth: GT文件路径。
        --detection_result: 结果json文件路径。

    验证精度数据

# 模型推理性能&精度(注：该模型只支持Batchsize=1的情况)<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 模型                    | batch_size | 官网pth精度  | 基准性能 | 310离线推理精度 | 310性能 | 310P离线推理精度 | 310P性能 |
|-------------------------|------------|--------------|----------|-----------------|---------|------------------|----------|
| faster_rcnn_r50_fpn_dcn |          1 | box AP:41.3% | 5.40FPS  | box AP:41.2%    | 4.61FPS | box AP:41.1%     | 8.00FPS  |
