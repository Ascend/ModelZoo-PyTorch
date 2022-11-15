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
  branch=master
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


  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
 

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。
 在已下载的源码包根目录下，执行如下命令
   ```
   git clone https://github.com/open-mmlab/mmdetection        
   cd mmdetection              
   git reset --hard 6c1347d7c0fa220a7be99cb19d1a9e8b6cbf7544        
   pip3 install -r requirements/build.txt     
   python3 setup.py develop
   cd ..                  
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）


  该模型使用COCO官网的COCO2017的5千张验证集进行测试，请用户需自行获取COCO2017数据集，上传数据集到mmdetection文件夹下，并重命名为data。目录结构如下：

   ```
  ├── GCNet      
  ├── mmdetection
    ├── data  
      ├── coco             
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行“GCNet_preprocess.py”脚本，完成预处理。

   ```
   python3 GCNet_preprocess.py --image_src_path=mmdetection/data/coco/val2017 --bin_file_path=val2017_bin --model_input_height=800 --model_input_width=1216 
   ```

   - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
    


   每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“val2017_bin”二进制文件夹。

   生成info文件。
     
   ```
   python3 gen_dataset_info.py jpg mmdetection/data/coco/val2017 coco2017_jpg.info
   ```
   - 参数说明：

           -   第一个参数：生成的数据集文件格式。
           -   第二个参数：原始数据文件的相对路径。
           -   第三个参数：生成的数据集文件保存的路径。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      从源码包中获取权重文件：“mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth”

   2. 导出onnx文件。

      i. 使用GCNet.diff对mmdetection源码进行修改


   ```
         cp GCNet.diff ./mmdetection/
         cd mmdetection
         patch -p1 < ./GCNet.diff
         cd ..
  ```

      ii. 修改环境下onnx源码，除去对导出onnx模型检查。
      

   ```
     vim /usr/local/python3.7.5/lib/python3.7/site-packages/torch/onnx/utils.py
   ```
    修改文件的_check_onnx_proto(proto)改为pass，执行:wq保存并退出。

         

      iii.使用“mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth”导出onnx文件。运行“mmdetection/tools/deployment/pytorch2onnx.py”脚本。

   ```
    python3 mmdetection/tools/deployment/pytorch2onnx.py mmdetection/configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py mmdetection/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth --output-file  GCNet.onnx --input-img mmdetection/demo/demo.jpg --test-img mmdetection/tests/data/color.jpg --shape 800 1216
   ```
   - 参数说明：

           -   --output-file：输出文件名。
           -   --input-img：输入图片。
           -   --test-img：输入测试图片。
           -   --shape：输入数据的大小。
        
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
         atc --framework=5 --model=GCNet.onnx --output=./GCNet_bs1 --input_shape="input:1,3,800,1216"  --log=error --soc_version=$\{chip\_name\}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
          

           运行成功后生成"GCNet_bs1.om"模型文件。

2. 开始推理验证

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
          python3 ais_infer.py --model /home/xhs/GCNet/GCNet_bs1.om --input /home/xhs/GCNet/val2017_bin --output /home/xhs/GCNet/result --batchsize 1 --outfmt BIN
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --input：bin文件路径。
             -   --output：输出结果路径。
               
        推理后的输出默认在当前目录result下。

        >**说明：** 
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   
   
   3. 精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据。
      ```
      python3 GCNet_postprocess.py --bin_data_path=./result/2022_11_03-17_26_58 --test_annotation=coco2017_jpg.info --det_results_path=detection-results --annotations_path=/opt/npu/coco/annotations/instances_val2017.json --net_out_num=3 --net_input_height=800 --net_input_width=1216
      ```
     
      
     - 参数说明：
    
               -   --bin_data_path：推理结果。
               -   --test_annotation：图片信息。
               -   --det_results_path：生成的后处理结果。
               -   --annotations_path：注释路径。
               -   --net_out_num：网络输出个数。
               -   --net_input_height：输入图片高度。
               -   --net_input_width：输入图片宽度。
                   
                   
                   
      ```
      python3 txt_to_json.py
      python3 coco_eval.py --ground_truth=mmdetection/data/coco/annotations/instances_val2017.json
      ```  

    - 参数说明：

               -   --ground_truth：标签数据。

  


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|    310       |      1            |    coco        |   0.610         |    8.364fps    |
|    310p        |     1           |         coco        |    0.610        |13.031fps  |
