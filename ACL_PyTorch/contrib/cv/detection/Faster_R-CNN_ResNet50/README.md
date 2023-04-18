# Faster R-CNN_ResNet50模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  ******

  

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Faster-R-CNN 在Fast RCNN的基础上使用RPN层代替Selective Search提取候选框，同时引入anchor box，大幅提高了two-stage检测网络的速度，向实时检测迈进。


- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn
  branch=master
  commit_id=a21eb25535f31634cef332b09fc27d28956fb24b
  model_name=faster_rcnn_r50_fpn
  ```
  



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据  | 数据类型 | 大小                        | 数据排布格式 |
  | :------: | :------: | :-------------------------: | :----------: |
  | input    | RGB_FP32 | batchsize x 3 x 1216 x 1216 | NCHW         |


- 输出数据

  | 输出数据  | 大小  | 数据类型 | 数据排布格式  |
  | :------: | :---: | :------: | :---------: |
  | boxes    | 100x5 | FLOAT32  | ND          |
  | labels   | 100   | INT64    | ND          |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 6.0.RC1 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.1   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓代码
   ```bash
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git 
   cd ./ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/detection/Faster_R-CNN_ResNet50/
   ```

   文件说明
   ```
   Faster_R-CNN_ResNet50
     ├── README.md                              # 此文档
     ├── coco_eval.py                           # 验证推理精度的脚本
     ├── get_info.py                            # 用于获取图像数据集的info文件
     ├── mmdetection.patch                      # 修改模型源码的patch文件
     ├── mmdetection_coco_postprocess.py        # 推理结果后处理脚本
     ├── mmdetection_coco_preprocess.py         # 数据集预处理脚本
     └── txt_to_json.py                         # 将推理结果txt文件转换为coco数据集评测精度的标准json格式
   ```

2. 安装依赖
   ```bash
   pip3 install -r requirements.txt

   # 安装mmpycocotools
   pip3 install mmpycocotools==12.0.3

   # 从源码安装mmcv-full
   git clone https://github.com/open-mmlab/mmcv.git
   cd mmcv
   git reset --hard 643009e4458109cb88ba5e669eec61a5e54c83be
   pip3 install -r requirements.txt
   MMCV_WITH_OPS=1 pip3 install -v -e .
   cd ..
   ```

   
   

3. 获取模型源码，并安装相应的依赖库

   ```bash
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   git reset --hard a21eb25535f31634cef332b09fc27d28956fb24b
   pip3 install -v -e .
   cd ..
   ```

4. 修改mmdetection源码

   使用mmdetection（v2.8.0）导出onnx前, 需要对源码做一定的改动，以适配Ascend NPU。

   ```bash
   patch -p0 < mmdetection.patch
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集和验证集

   该模型使用[COCO官网](https://cocodataset.org/#download)的coco2017的5千张验证集进行测试，图片与标签分别存放在```val2017/```与```annotations/instances_val2017.json```。

   ```bash
   wget http://images.cocodataset.org/zips/val2017.zip --no-check-certificate
   unzip -qo val2017.zip

   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip --no-check-certificate
   unzip -qo annotations_trainval2017.zip
   ```

2. 数据预处理
   将原始数据集转换为模型输入的数据。

   将原始数据（.jpeg）转化为二进制文件（.bin）。转化方法参考mmdetection预处理方法，以获得最佳精度。以coco_2017数据集为例，通过缩放、均值方差手段归一化，输出为二进制文件。

   执行mmdetection_coco_preprocess.py脚本，完成预处理。

   ```bash
   python3 mmdetection_coco_preprocess.py --image_folder_path val2017/ --bin_folder_path val2017_bin
   ```

   参数说明：
   - --image_folder_path: 图像数据集目录。
   - --bin_folder_path: 二进制文件输出目录。


3. JPG图片info文件生成


   后处理时需要输入数据集.jpg图片的info文件。使用get_info.py脚本，输入已经获得的图片文件,输出生成图片数据集的info文件。

   运行get_info.py脚本。

   ```bash
   python3 get_info.py jpg ./val2017/ coco2017_jpg.info
   ```
   参数说明：
   
   - 第一个参数为生成的数据集文件格式。
   - 第二个参数为coco图片数据文件的**相对路径**。
   - 第三个参数为生成的数据集信息文件保存的路径。
   

   运行成功后，在当前目录中生成```coco2017_jpg.info```。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       ```bash
       wget http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --no-check-certificate
       ```

   2. 修改cascade_rcnn_r50_fpn.py文件中nms_post参数 (可选)

   
      说明：

      > 由于NPU RoiExtractor算子的特殊性，适当减少其输入框的数量可以在小幅度影响精度的基础上大幅度提高性能，推荐将test_cfg中rpn层的nms_post参数从1000改为500，用户可以自行决定是否应用此项改动。

      打开文件。

      ```
      vim mmdetection/configs/_base_/models/cascade_rcnn_r50_fpn.py
      ```

      修改参数。

      ```bash
      test_cfg = dict(
          rpn=dict(
              nms_across_levels=False,
              nms_pre=1000,
              nms_post=500,    # Here
              max_num=1000,
              nms_thr=0.7,
              min_bbox_size=0),
          rcnn=dict(
              score_thr=0.05,
              nms=dict(type='nms', iou_threshold=0.5),
              max_per_img=100))
      ```


   3. 导出onnx文件。

      使用mmdetection/tools目录中的pytorch2onnx导出onnx文件。运行pytorch2onnx脚本。

      ```bash
      python3 mmdetection/tools/pytorch2onnx.py mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py ./faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --output-file faster_rcnn_r50_fpn.onnx --shape=1216
      ```

      参数说明：
   
      - 第一个参数为模型的配置文件。
      - 第二个参数为模型的权重文件。
      - --output-file: 生成的onnx模型文件保存路径。
      - --shape: 模型的输入shape。

      获得```faster_rcnn_r50_fpn.onnx```文件。

   4. 使用ATC工具将ONNX模型转OM模型。

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
         +--------------------------------------------------------------------------------------------+
         | npu-smi 22.0.0                       Version: 22.0.2                                       |
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 16.3         53                1236 / 1236           |
         | 0       0         | 0000:86:00.0    | 0            4060 / 21534                            |
         +===================+=================+======================================================+
         
         ```

      3. 执行ATC命令。
         ```
         atc --framework=5 --model=faster_rcnn_r50_fpn.onnx --output=faster_rcnn_r50_fpn --input_format=NCHW --input_shape="input:1,3,1216,1216" --log=debug --soc_version=${chip_name}
         ```

         参数说明：
         -   --model：为ONNX模型文件。
         -   --framework：5代表ONNX模型。
         -   --output：输出的OM模型。
         -   --input\_format：输入数据的格式。
         -   --input\_shape：输入数据的shape。
         -   --log：日志级别。
         -   --soc\_version：处理器型号。

         运行成功后生成```faster_rcnn_r50_fpn.om```模型文件。

2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  


   b.  执行推理。

```
python3 -m ais_bench --model faster_rcnn_r50_fpn.om --input=val2017_bin --output=result
```
  -   参数说明：
       -   model：om文件路径。
       -   input：输入文件。
       -   output：输出文件所存目录。

  推理后的输出默认在当前目录result下。

   c.  精度验证。

 本模型提供后处理脚本，将二进制数据转化为txt文件，执行脚本。

```
python3 mmdetection_coco_postprocess.py --bin_data_path=result/${infer_result_dir} --prob_thres=0.05 --det_results_path=detection-results --test_annotation=coco2017_jpg.info
```

- 参数说明：

   -   bin_data_path：推理输出目录 (注意替换成实际目录，如```2022_12_16-18_01_01/```)。

   -   prob_thres：框的置信度阈值。

   -   det_results：后处理输出目录。

评测结果的mAP值需要使用官方的pycocotools工具，首先将后处理输出的txt文件转化为coco数据集评测精度的标准json格式。

执行转换脚本。

```
python3 txt_to_json.py --npu_txt_path detection-results --json_output_file coco_detection_result
```
- 参数说明：

   -   --npu_txt_path: 输入的txt文件目录。

   -   --json_output_file: 输出的json文件路径。


运行成功后，生成```coco_detection_result.json```文件。
调用coco_eval.py脚本，输出推理结果的详细评测报告。

```
python3 coco_eval.py --detection_result coco_detection_result.json --ground_truth=annotations/instances_val2017.json
```
- 参数说明：
   - --detection_result：推理结果json文件。

   - --ground_truth：```instances_val2017.json```的存放路径。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 | 基准性能 |
| :------: | :---------: | :-----: | :---: | :--: | :--: |
| Ascend310P | 1 | coco2017 | 37.2 | 15.759 | 19.4 |