# Cascade_RCNN_R101模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Cascade R-CNN是目标检测two-stage算法的代表之一，使用cascade回归作为一种重采样的机制，逐stage提高proposal的IoU值，从而使得前一个stage重新采样过的proposals能够适应下一个有更高阈值的stage。论文作者通过观察目标检测中正样本选取时的IOU不同时分类器和回归器的不同表现，设计了级联的分类回归结构，取得了很大的mAp提升。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection.git
  branch=v2.8.0
  commit_id=a21eb25535f31634cef332b09fc27d28956fb24b
  model_name=Cascade_RCNN_fpn
  ```


  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {url}        # 克隆仓库的代码
  cd mmdetection              # 切换到模型的代码仓目录
  git checkout {branch}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | --------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 1216 x 1216 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  |  boxes  | 100 x 5 | FLOAT32  | ND           |
  | labels  | 100 | INT64  | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    |
| ------------------------------------------------------------ | ------- |
| CANN                                                         | 5.1.RC2 |
| Python                                                       | 3.7.5   |
| PyTorch                                                      | 1.7.0   |
| torchvision                                                  | 0.8.0   |
| mmcv-full                                                    | 1.2.5   |
| mmpycocotools                                                | 12.0.3  |
| onnx                                                         | 1.12.0  |
| protobuf                                                     | 3.20.1  |
| onnxoptimizer                                                | 0.3.0   |
| onnxruntime                                                  | 1.5.2   |
| opencv-python                                                | 4.4.0.46|
| numpy                                                        | 1.21.5  |

特殊安装：
- pycocotools安装
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>



## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
    
    推理测试使用coco2017验证数据集，一共5000张图片数据，以及json标签文件instances_val2017.json。
    数据集和标签文件请自行获取。
    数据集解压后位于val2017文件夹中，在源码包下创建data目录，并置于data目录下。

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   1.将原始数据集转换为模型输入的二进制数据。

    将原始数据（.jpg）转化为二进制文件（.bin）。转化方法参考mmdetection预处理方法，以获得最佳精度。以coco_2017数据集为例，通过缩放、均值方差手段归一化，输出为二进制文件。
    
   2.执行Cascade_RCNN_R101_preprocess.py脚本。
    ```
    python3.7 Cascade_RCNN_R101_preprocess.py --image_folder_path ./data/val2017 --bin_folder_path ./data/val2017_bin
    ```
    - 参数说明：
      - --image_folder_path：原始数据验证集（.jpg）所在路径。
      - --bin_folder_path：输出的二进制文件（.bin）所在路径。

    每个图像对应生成一个二进制文件。

   3.二进制输入info文件生成
    使用脚本计算精度时需要输入二进制数据集的info文件
    使用get\_info.py脚本，输入已经获得的图片文件，输出生成图片数据集的info文件。
    
    运行get_info.py脚本。
    ```
    python3.7 get_info.py jpg ../data/val2017 coco2017_jpg.info
    ```
    第一个参数为生成的数据集文件格式，第二个参数为coco图片数据文件的相对路径，第三个参数为生成的数据集信息文件保存的路径。运行成功后，在当前目录中生成coco2017_jpg.info。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [Cascade_RCNN_R101预训练pth权重文件](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth )
      下载经过训练的Cascade-RCNN-R101-FPN-1X-COCO模型权重文件，并移动到Modelzoo源码包中。

   2. mmdetection代码下载和迁移，执行命令。
      开源仓库代码下载，在源码包目录下运行命令
      ```
      git clone --branch v2.8.0 https://github.com/open-mmlab/mmdetection
      cd mmdetection
      git reset --hard a21eb25535f31634cef332b09fc27d28956fb24b
      pip install -v -e .
      ```

      运用补丁修改代码。
      ```
      patch -p1 < ../Cascade_RCNN_R101.patch
      cd ..
      ```


   3. 导出onnx文件。

       使用mmdetection提供的pytorch2onnx.py导出onnx文件。

       运行脚本。

       ```
       python mmdetection/tools/pytorch2onnx.py mmdetection/configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py ./cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth --output-file=cascade_rcnn_r101.onnx --shape 1216 --show
       ```

       获得XXX.onnx文件。

   4. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

        ```
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        ```

        > 说明：
        > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
        

      2. 执行命令查看芯片名称（$\{chip\_name\}），确保device空闲  。

        ```
        npu-smi info
        #该设备芯片名为Ascend310P3 （自行替换）
        回显如下：
        ```

~~~
+--------------------------------------------------------------------------------------------+
| npu-smi 22.0.0                       Version: 22.0.2                                       |
+-------------------+-----------------+------------------------------------------------------+
| NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
| Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
+===================+=================+======================================================+
| 0       310P3     | OK              | 16.7         57                0    / 0              |
| 0       0         | 0000:5E:00.0    | 0            932  / 21534                            |
+===================+=================+======================================================+
~~~


      3. 执行ATC命令。

        ```
        atc --framework=5 --model=./cascade_rcnn_r101.onnx --output=cascade_rcnn_r101 --input_format=NCHW --input_shape="input:1,3,1216,1216" --soc_version=$\{chip\_name\} --out_nodes="Concat_947:0;Reshape_949:0" --log info
        ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成cascade_rcnn_r101.om模型文件。


2. 开始推理验证。
  
    a. 创建结果保存目录。

        ```
        mkdir ais_infer_result
        ```
  
    b.  执行推理。

        ```
        python3.7 ais_infer.py --model cascade_rcnn_r101_aoe.om --input data/val2017_bin/ --output ais_infer_result --outfmt BIN --batchsize 1
        ```

      - 参数说明：
        - --input：bin数据目录。
        - --model：om文件路径。
        - --output：输出结果的目录。
        - --outfmt：输出结果的格式。
        - --batchsize：推理batchsize大小，只支持batchsize=1。
      
      注意模型和数据集路径，可以用绝对路径，推理结果保存到了ais\_infer\_result下自动创建的日期文件夹中。
      
    > **说明：** 
    > 执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

    c.  精度验证。
   
      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在ais_infer_detection_result.json中，可视化结果保存在ais_infer_detection_result中。

      ```
      python3 Cascade_RCNN_R101_postprocess.py --bin_data_path=ais_infer_result/日期文件夹 --prob_thres=0.05 --ifShowDetObj --det_results_path=ais_infer_detection_results --test_annotation=coco2017_jpg.info --json_output_file ais_infer_detection_result --detection_result ais_infer_detection_result.json 
      ```

- 参数说明：

  + --ais_infer_result/日期文件夹： 推理输出的bin文件路径
  + --coco2017_jpg.info：测试图片信息
  + --ais_infer_detection_results：生成推理结果所在路径
  + --ais_infer_detection_result.json：生成结果文件
    

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>


| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|   310*4   |    1       |   cocoVal2017    |  0.420    |   6.07484   |
|   310P    |    1       |   cocoVal2017    |  0.419    |   9.14181   |
|   T4      |    1       |   cocoVal2017    |  0.420    |   4.3   |