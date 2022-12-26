# FasterRCNN-DCN模型-推理指导


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
  model_name=FasterRCNN-DCN子模型，其配置文件为url路径下的faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py
  ```

 


  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | 1 x 3 x 1216 x 1216 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | boxes  | 100 × 5 | FLOAT32  | ND           |
  | labels  | 100 × 1 | INT64  | ND           |




# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

1. 下载本模型代码包，并上传至服务器解压至用户目录下


2. 安装常规依赖。

   ```
   pip3.7 install -r requirment.txt
   conda install decorator
   conda install sympy
   ```
3. 安装mmcv。(注：此步骤安装时间较长，约10分钟左右，请耐心等候)
   ```
   git clone https://github.com/open-mmlab/mmcv -b master
   cd mmcv
   git checkout v1.2.4
   MMCV_WITH_OPS=1 pip install -e .
   patch -p1 < ../mmcv.patch
   cd ..
   ```
4. 安装mmdetection。(注：此步骤安装时间较长，约5分钟左右，请耐心等候)
   ```
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   git reset --hard a21eb25535f31634cef332b09fc27d28956fb24b
   pip install -v -e .
   cd ..
   ```
   注：mmcv与mmdetection均安装在本模型代码文件夹下。其目录结构如下所示
   ```
    |--FasterRCNN-DCN
        |--mmcv             //mmcv文件夹
        |--mmdetection      //mmdetection文件夹
        |--其他文件以及文件夹
   ```
5. 修改mmdetection源码适配Ascend NPU

   使用mmdetection（v2.8.0）导出onnx前, 需要对源码做一定的改动，以适配Ascend NPU。具体的代码改动请参考Modelzoo源码包中的Faster_RCNN_DCN修改实现.md文档，修改后的同名文件已在源码包中提供，用户可以直接在相应目录中备份原文件并替换。 

   **（注意替换文件前请自行备份源文件）**
   ```
   cp ./pytorch_code_change/bbox_nms.py ./mmdetection/mmdet/core//post_processing/bbox_nms.py
   cp ./pytorch_code_change/rpn_head.py ./mmdetection/mmdet/models/dense_heads/rpn_head.py
   cp ./pytorch_code_change/single_level_roi_extractor.py ./mmdetection/mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py
   cp ./pytorch_code_change/delta_xywh_bbox_coder.py ./mmdetection/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py
   ```
   - 说明：
   
        - 在bbox_nms.py文件中用NPU算子BatchMultiNMS代替原mmdetection中的NMS层算子，替换后精度无损失。同时等价换一个expand算子，使导出的onnx中不含动态shape。

        - 在rpn_head.py 文件中用NPU算子BatchMultiNMS代替原mmdetection中的NMS层算子，替换后精度无损失。

        - single_level_roi_extractor.py 中用NPU算子RoiExtractor代替原mmdetection中的RoiAlign层算子，替换后精度无损失。

        - delta_xywh_bbox_coder.py 中修改坐标的轴顺序，使切片操作在NPU上效率更高，整网性能提升约7%；修改means和std计算方法使其表现为固定shape。

        - 由于框架限制，当前模型仅支持batchsize=1的场景
   
    
6. 屏蔽掉torch.onnx中的model_check相关代码。

   注册添加NPU自定义算子后需要手动屏蔽掉torch.onnx中的model_check相关代码，否则导出onnx过程中无法识别自定义算子会导致报错。 pytorch安装位置查找命令如下。
   ```
   pip3.7 show torch
   ```
   返回pytorch安装位置（如：xxx/lib/python3.7/site-packages）。打开文件改路径下的/torch/onnx/utils.py文件。搜索_check_onnx_proto(proto)并注释代码，添加pass代码，后保存并退出。
   ```
   if enable_onnx_checker and \
    operator_export_type is OperatorExportTypes.ONNX and \
        not val_use_external_data_format:
    # Only run checker if enabled and we are using ONNX export type and
    # large model format export in not enabled.
    # _check_onnx_proto(proto)
    pass
   ```
7. 修改./mmdetection/configs/_ base _/models/faster_rcnn_r50_fpn.py文件中nms_post参数。将其值由1000修改为500
   ```
   test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=500,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
   ```
    说明：
    
    由于NPU RoiExtractor算子的特殊性，适当减少其输入框的数量可以在小幅度影响精度的基础上大幅度提高性能，推荐将test_cfg中rpn层的nms_post参数从1000改为500，用户可以自行决定是否应用此项改动。



## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型推理数据集采用coco_val_2017，请用户自行获取coco_val_2017数据集。将instances_val2017.json文件和val2017文件夹上传并解压数据集到ModelZoo的源码包路径下。其目录结构如下所示（注：该数据集在服务器上的路径为/opt/npu/coco/）
   ```
    |--FasterRCNN-DCN
        |--instances_val2017.json   //验证集标注信息
        |--val2017                  //验证集文件夹
        |--其他文件以及文件夹
   ```

2. 数据预处理。

   2.1：执行mmdetection_coco_preprocess.py，将数据集转换(.jpg)为二进制数据(.bin)文件。
   ```
    python3.7 mmdetection_coco_preprocess.py --image_folder_path ./val2017 --bin_folder_path val2017_bin
   ```
   - 参数说明：
        
        - --image_folder_path：原始数据验证集（.jpg）所在路径。

        - --bin_folder_path：输出的二进制文件（.bin）所在路径。
    
    成功运行后生成val2017_bin文件夹
    
    2.2：执行get_info.py，以val2017文件夹的jpg图片生成coco2017_jpg.info文件
   ```
   python3.7 get_info.py jpg ./val2017 coco2017_jpg.info
   ```
     - 参数说明：
       -   参数1(get_info.py)：执行的Python文件
        
       -   参数2(jpg)：输入数据的数据格式
        
       -   参数3(./val2017)：输入数据的文件夹路径
        
       -   参数4(coco2017_jpg.info)：生成的info信息文件
    
       成功运行后生成coco2017_jpg.info文件





## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
        点击[此链接](https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco_20200203-4f85c69c.pth)
        下载经过训练的faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco模型权重文件，并移动到Modelzoo源码包中。
       

   2. pth导出onnx文件。

      1. 生成onnx模型。
         
            调用mmdete/tools目录中的pytorch2onnx脚本生成onnx模型。这里注意指定shape为1216。当前框架限制，仅支持batchsize=1的场景。

         ```
         python3.7 mmdetection/tools/pytorch2onnx.py mmdetection/configs/dcn/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py ./faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco_20200203-4f85c69c.pth --output-file=FasterRCNNDCN.onnx --shape=1216 --verify --show
         ```

         获得FasterRCNNDCN.onnx文件。

      2. 优化onnx模型。
            
            上述得到的onnx模型在转换om模型的过程中，会因部分结点的多个输入数据的数据类型不一致而导致转换失败。需要通过以下代码来查找和修改这部分结点。

         ```
         python3.7 modifyonnx.py --batch_size=1
         ```
         获得FasterRCNNDCN_change_bs1.onnx文件。

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
         | npu-smi 22.0.0                       Version:22.0.2                                        |
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 16.2         42                0    / 0              |
         | 0       0         | 0000:86:00.0    | 0            994  / 21534                            |
         +===================+=================+======================================================+
         ```

      3. 执行ATC命令。

         ```
         atc --model=FasterRCNNDCN_change_bs1.onnx --framework=5 --output=FasterRCNNDCN --input_format=NCHW --input_shape="input:1,3,1216,1216" --log=info --soc_version=Ascend310P3
         ```
         
        - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
    
           运行成功后生成FasterRCNNDCN.om模型文件。



2. 开始推理验证。

a.  安装ais包，激活环境。（安装ais_bench推理工具，以下安装包为linux版本）
    请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。  

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

b.  执行推理。(提前将官网上的ais_bench源码包放入项目文件夹中,其目录结构如下。注：如果导入的是ais_infer的父包tool工具包，则需要对代码中的参数1添加对应的目录结构)
   ```
    |--FasterRCNN-DCN
        |--ais_infer            //ais_bench推理工具的源码包
        |--其他文件以及文件夹
   ```

    mkdir ais_results
    python3.7 -m ais_bench --model ./FasterRCNNDCN.om --input ./val2017_bin --output ./ais_results --outfmt BIN --batchsize 1
    

-   参数说明：
    
    -   --model：om模型路径
    -   --input：执行推理所输入的bin文件夹
    -   --output：输出的推理结果的文件夹
    -   --outfmt：输出推理结果的格式
		

        推理后的输出默认在当前目录的./ais_results文件夹下。
c. 修改输出文件名。

>**说明：**
为适配精度验证的后处理代码，需要将上述输出文件重命名。重命名代码可以集成到ais_infer.py或新编写一个py文件进行处理。考虑到用户复现代码过程所使用的ais源码包同步与官网，不便于修改。故基于解耦的思想，新增了modifyname.py文件用作修改。其执行命令如下。


    python3.7 modifyname.py --aisbin_path ./ais_results/2022_08_11-10_21_35
    
-   参数说明：

    -   -aisbin_path：上述b中生成的文件夹，其中“2022_08_11-10_21_35”请用户结合自己的生成时间进行替换。


d.  精度验证。
-   bin转txt：
    ```
    python3.7 mmdetection_coco_postprocess.py --bin_data_path=./ais_results/2022_08_11-10_21_35 --prob_thres=0.05 --ifShowDetObj --det_results_path=detection-results --test_annotation=coco2017_jpg.info
    ```
    -   参数说明：
        
           --bin_data_path：需要转换为txt信息的bin文件夹，其中“2022_08_11-10_21_35”请用户结合自己的生成时间进行替换。
    
    生成detection-results文件夹
-   txt转json：
    ```
    python3.7 txt_to_json.py
    ```
    生成coco_detection_result.json文件
-   json对比获取精度数据：
    ```
    python3.7 coco_eval.py
    ```
    验证精度数据

# 模型推理性能&精度(注：该模型只支持Batchsize=1的情况)<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ------| ----------- | ------- | ------- |
|    310    |   1   |coco_val_2017|   0.444 |  1.599  |
|    310P3  |   1   |coco_val_2017|   0.443 |  2.478  |
|    T4     |   1   |coco_val_2017|   0.445 |  4.4    |