# Cascade_RCNN_R101模型-推理指导


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

Cascade R-CNN是目标检测two-stage算法的代表之一，使用cascade回归作为一种重采样的机制，逐stage提高proposal的IoU值，从而使得前一个stage重新采样过的proposals能够适应下一个有更高阈值的stage。论文作者通过观察目标检测中正样本选取时的IOU不同时分类器和回归器的不同表现，设计了级联的分类回归结构，取得了很大的mAp提升。




- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection.git
  commit_id=a21eb25535f31634cef332b09fc27d28956fb24b
  code_path=contrib/cv/detection/Cascade_RCNN_R101_FPN
  ```
  
 


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 1216 x 1216 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | boxes  | FLOAT32  | 100 x 5 | ND           |
  | labels  | INT64       | 100 x 1 | ND           |
 
# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17(NPU驱动固件版本为6.0.RC1)  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
                                               



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。

   ```
   pip install -r requirements.txt     
   ```
>**说明：** 
>torch1.7在arm上不支持pip直接安装，如在arm上复现请参考[官方源码编译步骤](https://github.com/pytorch/pytorch/tree/v1.7.0#installation)安装

2. 获取源码。
    1. 安装开源仓
   ```
   git clone --branch v2.8.0 https://github.com/open-mmlab/mmdetection
   cd mmdetection
   git reset --hard a21eb25535f31634cef332b09fc27d28956fb24b
   pip install -v -e .

   ```
    2. 修改模型
   ```
   patch -p1 < ../Cascade_RCNN_R101.patch
   cd ..
   ```

3. 安装mmcv-full,mmpycocotools

   ```
   pip install openmim
   mim install mmcv-full==1.2.4
   mim install mmpycocotools==12.0.3
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。
   本模型已在coco 2017数据集上验证过精度。推理数据集采用coco_val_2017，请用户自行获取coco_val_2017数据集。将instances_val2017.json文件和val2017文件夹按照如下目录结构上传并解压数据集到服务器任意目录。
    最终，数据的目录结构如下：
   ```
   ├── coco
       ├── val2017   
       ├── annotations
            ├──instances_val2017.json
         

   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。


   执Cascade_RCNN_R101_preprocess.py脚本，完成预处理。

   ```
   python Cascade_RCNN_R101_preprocess.py --image_folder_path ./coco/val2017 --bin_folder_path ./val2017_bin
   ```
   - 参数说明：
      -  --image_folder_path：数据集路径。
      -  --bin_folder_path：预处理后的数据文件的相对路径。
      
    
    运行成功后，会在当前目录下生成二进制文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       [cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Cascade_RCNN_R101_FPN/PTH/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth)

   2. 导出onnx文件。

      1. 使用mmdetection/tools/pytorch2onnx.py导出onnx文件。

      
  

         ```
         python mmdetection/tools/pytorch2onnx.py     mmdetection/configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py                   ./cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth --output-file=cascade_rcnn_r101.onnx --shape 1216 

         ```
         - 参数说明：
            -  --shape : 模型大小
            -  --output-file: 输出onnx模型
          

         获得cascade_rcnn_r101.onnx文件,模型只支持bs1。



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
            atc --framework=5\ 
                 --model=./cascade_rcnn_r101.onnx\ 
                 --output=./cascade_rcnn_r101\ 
                 --input_format=NCHW\ 
                 --input_shape="input:1,3,1216,1216"\ 
                 --log=error\
                 --out_nodes="Concat_947:0;Reshape_949:0"\
                 --soc_version=Ascend${ChipName}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --out_nodes: 输出节点

        运行成功后生成cascade_rcnn_r101.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        
    ```
    python -m ais_bench --model ./cascade_rcnn_r101.om\  
                                  --input ./val2017_bin/\ 
                                  --output ./\ 
                                  --batchsize 1\
                                  --outfmt BIN\
                                  --output_dirname result
    ```

    - 参数说明：

      - --model: OM模型路径。
      - --input: 存放预处理bin文件的目录路径
      - --output: 存放推理结果的目录路径
      - --batchsize：每次输入模型的样本数
      - --outfmt: 推理结果数据的格式
      - --output_dirname: 输出结果子目录
        推理后的输出默认在当前目录result下。


   3. 精度验证。

      运行get_info.py,生成图片数据文件
    ```
    python get_info.py jpg ./coco/val2017 coco2017_jpg.info
    ```
    - 参数说明：

      - --第一个参数：原始数据集
      - --第二个参数：图片数据信息

      调用“Cascade_RCNN_R101_postprocess.py”评测模型的精度。

    ```
    python Cascade_RCNN_R101_postprocess.py --bin_data_path=result --prob_thres=0.05 --ifShowDetObj --det_results_path=detection-results --test_annotation=coco2017_jpg.info --img_path ./coco/val2017 --ground_truth ./coco/annotations/instances_val2017.json

    ```
    - 参数说明：

      - --bin_data_path: 推理结果。
      - --test_annotatio: 原始图片信息文件。
      - --det_results_path: 后处理输出结果。
      - --ifShowDetObj：是否将box画在图上显示。
      - --prob_thres: 目标框的置信度阈值
      - --img_path: 原始数据
      - --ground_truth: 数据标签
    
   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

    ```
    python -m ais_bench --model ./cascade_rcnn_r101.om --loop 100 --batchsize 1
    ```

    - 参数说明：

      - --model: om模型
      - --batchsize: 每次输入模型样本数
      - --loop: 循环次数    



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

1. 精度对比

    | Model       | batchsize | Accuracy | 
    | ----------- | --------- | -------- |
    | Cascade_rcnn_r101| 1       | bbox_mAP = 0.419 |

2. 性能对比

    | batchsize | 310 性能 | 310P 性能 | 
    | ---- | ---- | ---- | 
    | 1 | 6.1 |9.1|
    