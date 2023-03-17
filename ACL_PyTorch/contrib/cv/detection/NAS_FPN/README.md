# NAS_FPN模型-推理指导


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
在目标检测中，不同尺度的特征在建模语义信息和细节信息上具有不同的表现，因此对多尺度的特征进行融合对于提升检测效果至关重要。NAS-FPN自动的对自顶向下和自底向上的双向融合策略进行搜索，从而得到优于FPN和PANet的融合策略。NAS-FPN 与 RetinaNet 框架中的若干骨干模型相结合，实现了优于当前最佳目标检测模型的准确率和延迟权衡。该架构将移动检测准确率提高了 2 AP。

<u>***简单描述模型的结构、应用、优点等信息。***</u>


- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection
  commit_id=a21eb25535f31634cef332b09fc27d28956fb24b
  ```



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 640 x 640 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | boxes    | FLOAT32  | 100 x 5  | ND           |
  | labels   | FLOAT32  | 100      | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.7.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

    ```
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection  
    git reset a21eb25535f31634cef332b09fc27d28956fb24b --hard
    pip3 install -v -e .
    patch -p1 < ../NAS_FPN.patch   
    cd ..
    ```

2. 安装依赖。
    ```
    pip3 install -r requirements.txt
    pip3 install openmim
    mim install mmcv-full==1.2.4
    mim install mmpycocotools==12.0.3
    ```

    通过命令找到mmcv-full安装位置。
    ```shell
    pip3 show mmcv-full
    ```
    
    修改mmcv中的算子脚本，使其支持导出onnx
    ```shell
    patch -p0 xxx/lib/python3.7/site-packages/mmcv/ops/merge_cells.py merge_cells.patch
    ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

    本模型支持coco2017验证集。获取数据集后，将annotations文件和val2017文件夹解压并上传数据集到源码包路径下。目录结构如下：

    ```
    NAS_FPN
    ├── coco      
        ├── annotations
        └── val2017
    ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

    执行mmdetection_coco_preprocess脚本，完成预处理。

    ```
    python3 mmdetection_coco_preprocess.py --image_folder_path ./coco/val2017 --bin_folder_path val2017_bin 
    ```
    - 参数说明：
        - --image_folder_path：原始数据验证集图片所在路径。
        - --bin_folder_path：输出的二进制文件所在路径。
        
    运行成功后在主目录下得到val2017_bin文件夹。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       [NAS_FPN权重文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/NAS-FPN/PTH/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth)

   2. 导出onnx文件。

        使用mmdetection/tools中的pytorch2onnx.py导出onnx文件。

        ```shell
        python3 mmdetection/tools/pytorch2onnx.py mmdetection/configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth --output-file=nas_fpn.onnx --shape=640
        ```
        - 参数说明：
            - 第一个参数为配置文件路径。
            - 第二个参数为权重文件路径。
            - --shape：输入数据大小。
            - --output-file：转出的onnx模型路径。

        获得nas_fpn.onnx文件。


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

        ```shell
        atc --model=./nas_fpn.onnx \
            --framework=5 \
            --output=nas_fpn_bs1 \
            --input_format=NCHW \
            --input_shape="input:1,3,640,640" \
            --log=error \
            --soc_version=Ascend${chip_name}
        ```

         - 参数说明：

           -  --model：为ONNX模型文件。
           -  --framework：5代表ONNX模型。
           -  --output：输出的OM模型。
           -  --input\_format：输入数据的格式。
           -  --input\_shape：输入数据的shape。
           -  --log：日志级别。
           -  --soc\_version：处理器型号。

           运行成功后生成nas_fpn_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```shell
        python3 -m ais_bench --model nas_fpn_bs1.om --input val2017_bin --output ./ --output_dirname result
        ```

        -  参数说明：

            -  model：模型路径。
            -  input：模型的输入，预处理生成的文件路径。
            -  output：模型输出目录。
            -  output_dirname：模型的输出子目录。

        推理后的输出默认在当前目录result下。


   3. 精度验证。
      - 运行get_info.py脚本，生成图片数据info文件。

         ```shell
         python get_info.py jpg ./coco/val2017 val2017.info
         ```

        - 参数说明：
            - 第一个参数为生成的数据集文件格式
            - 第二个参数为原始数据文件相对路径
            - 第三个参数为生成的info文件名

         运行成功后，在当前目录生成val2017.info。

      - 执行后处理脚本，将二进制数据转化为txt文件：
         ```shell
         python mmdetection_coco_postprocess.py \
               --bin_data_path=./result \
               --test_annotation=val2017.info \
               --det_results_path=detection-results
         ```

        - 参数说明：
            - --bin_data_path: 推理结果所在路径。
            - --test_annotation: 原始图片信息文件。
            - --det_results_path：后处理输出目录。

      - 执行转换脚本，将txt文件转化为json文件：
        ```shell
        python3 txt_to_json.py --npu_txt_path=detection-results --json_output_file=coco_detection_result
        ```

        - 参数说明：
            - --npu_txt_path: txt文件目录。
            - --json_output_file: 生成的json文件。

      - 调用`coco_eval.py`脚本，输出精度报告：
        ```shell
        python3 coco_eval.py --ground_truth=coco/annotations/instances_val2017.json --detection_result=coco_detection_result.json
        ```

        - 参数说明：
            - --ground_truth: coco数据集标准文件。
            - --detection_result: 模型推理结果文件。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7 -m ais_bench --model=${om_model_path} --loop=20
        ```

      - 参数说明：
        - --model：om模型路径。
        - --loop：推理次数。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度 | 性能 |
| --------- | ------------ | ---------- | ------------ | --------------- |
|   310P3   |       1      |   COCO2017 |  map: 0.404  |      72.68      |

仅支持batch size为1