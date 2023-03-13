# GCNet模型-推理指导


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

GCNet最初在arxiv中被提出。结合Non-Local Networks (NLNet)和Squeeze-Excitation Networks (SENet)的优点，GCNet为全局上下文建模提供了一种简单、快速和有效的方法，在各种识别任务的主要基准上通常优于NLNet和SENet。




- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet
  commit_id=f08548bfd6d394a82566022709b5ce9e6b0a855e
  code_path=contrib/cv/segmentation/GCNet
  model_name=gcnet
  ```
  
 


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 800 x 1216 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | boxes  | FLOAT32  | batchsize x100 x 5 | ND           |
  | labels |  INT64   | batchsize x 100 x 1  | ND        |
  | masks  |  FLOAT32|  batchsize x 100 x 80 x 28 x 28   | ND  |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
                                               



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

2. 获取源码。

   ```
   git clone https://github.com/open-mmlab/mmdetection        
   cd mmdetection              
   git reset --hard 6c1347d7c0fa220a7be99cb19d1a9e8b6cbf7544          
   python3 setup.py develop
   cd ..        
   ```

3. 源码改动。

   i. 使用GCNet.diff对mmdetection源码进行修改
   ```
   cp GCNet.diff ./mmdetection/
   cd mmdetection
   patch -p1 < ./GCNet.diff
   cd ..
   ```
   ii. 修改环境下onnx源码，除去对导出onnx模型检查。
   ```
   进入 python依赖安装路径/torch/onnx/utils.py，修改文件的_check_onnx_proto(proto)改为pass，执行:wq保存并退出。

   ```

4. 安装mmcv-full,mmpycocotools

   ```
   pip3 install openmim
   mim install mmcv-full==1.2.5
   mim install mmpycocotools==12.0.3
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。
   本模型已在coco 2017数据集上验证过精度。推理数据集采用coco_val_2017，请用户自行获取coco_val_2017数据集。将instances_val2017.json文件和val2017文件夹按照如下目录结构上传并解压数据集到Retinanet。
    最终，数据的目录结构如下：
   ```
   ├──datasets
      |── coco
       |──annotations
           |──instances_val2017.json    //验证集标注信息       
       |── val2017                      // 验证集文件夹

   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。
   运行数据预处理脚本，将原始数据转换为符合模型输入要求的bin文件。
   ```shell
   python3 GCNet_preprocess.py --image_src_path=./datasets/coco/val2017 --bin_file_path=val2017_bin --model_input_height=800 --model_input_width=1216
   ```
    -参数说明：
     + --image_src_path：原始数据验证集（.jpg）所在路径。
     + --bin_file_path：输出的二进制文件（.bin）所在路径。
     + --model_input_height：模型输入图像高度像素数量。
     + --model_input_width：模型输入图像宽度像素数量。
    
    运行成功后，会在当前目录下生成二进制文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       该推理项目使用权重文件[mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/GCNet/PTH/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth)

   2. 导出onnx文件。

      1. 运行“mmdetection/tools/deployment/pytorch2onnx.py”脚本。 


         ```
         python3 mmdetection/tools/deployment/pytorch2onnx.py mmdetection/configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py mmdetection/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth --output-file  GCNet.onnx --input-img mmdetection/demo/demo.jpg --test-img mmdetection/tests/data/color.jpg --shape 800 1216
         ```
         -参数说明：
          + --output-file：输出文件名。
          + --input-img：输入图片。
          + --test-img：输入测试图片。
          + --shape：输入数据的大小。
          
         获得“GCNet.onnx”文件，该模型只支持batchsize1。


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
         atc --framework=5 --model=GCNet.onnx --output=./GCNet_bs1 --input_shape="input:1,3,800,1216"  --log=error --soc_version=${chip_name} --input_format=NCHW 
          
         ```

         - 参数说明：
            + --model: ONNX模型文件所在路径。
            + --framework: 5 代表ONNX模型。
            + --input_format: 输入数据的排布格式。
            + --input_shape: 输入数据的shape。
            + --output: 生成OM模型的保存路径。
            + --log: 日志级别。
            + --soc_version: 处理器型号。

    运行成功后生成GCNet_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
        python3 -m ais_bench --model ./GCNet_bs1.om --input ./val2017_bin --output ./ --batchsize 1 --outfmt BIN --output_dirname result
        ```

      - 参数说明：
        + --model: OM模型路径。
        + --input: 存放预处理bin文件的目录路径
        + --output: 存放推理结果的目录路径
        + --batchsize：每次输入模型的样本数
        + --outfmt: 推理结果数据的格式
        + --output_dirname: 推理结果输出子文件夹

        推理后的输出默认在当前目录result下。


   3. 精度验证。

      运行gen_dataset_info.py脚本，生成图片数据info文件。


        ```
        python3 gen_dataset_info.py jpg ./datasets/coco/val2017 coco2017_jpg.info
        ```

        - 参数说明：
          + 第一个参数为生成的数据集文件格式
          + 第二个参数为原始数据文件相对路径
          + 第三个参数为生成的info文件名

      运行成功后，在当前目录生成coco2017_jpg.info
     执行后处理脚本，计算精度：
      ```
      python3 GCNet_postprocess.py --bin_data_path=./result --test_annotation=coco2017_jpg.info --det_results_path=detection-results --annotations_path=./datasets/coco/annotations/instances_val2017.json --net_out_num=3 --net_input_height=800 --net_input_width=1216
      ```
    
      - 参数说明：
        + --bin_data_path: 推理结果所在路径
        + --annotations_path: 注释路径
        + --test_annotation: 原始图片信息文件
        + --det_results_path: 后处理输出结果
        + --net_out_num: 网络输出个数
        + --net_input_height: 网络高
        + --net_input_width: 网络宽

                 
      ```
      python3 txt_to_json.py
      python3 coco_eval.py --ground_truth=./datasets/coco/annotations/instances_val2017.json
      ```  

         - 参数说明：
            + --ground_truth：标签数据。
       
   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
        python3 -m ais_bench --model GCNet_bs1.om --loop 100 --batchsize 1
        ```

      - 参数说明：
        + --model: om模型
        + --batchsize: 每次输入模型样本数
        + --loop: 循环次数    



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|    310       |      1            |    coco        |   0.610         |    8.364fps    |
|    310p        |     1           |         coco        |    0.610        |13.031fps  |
