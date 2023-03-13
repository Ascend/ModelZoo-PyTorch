# HRNet模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

 HRNet是针对人体关键点检测提出的网络，一次只能检测一个个体.

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmpose
  commit_id=4c397f2db99cc6dbf0787ec6a44266eddd231bb3
  code_path=https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/HRNet_mmpose
  model_name=HRNet
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小            | 数据排布格式 |
  | -------- |---------------| ------------------------- | ------------ |
  | input    | RGB_FP32 | 1 x 3 x H x W | ND         |
注:HRNet模型推理阶段的数据处理，短轴固定为512，长轴不固定

- 输出数据

  | 输出数据 | 数据类型 | 大小               | 数据排布格式 |
  | -------- |------------------| -------- | ------------ |
  | output1  | FLOAT32  | 1 x 34 x H‘ x W’ | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本      | 环境准备指导                                                 |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.10.1  | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/open-mmlab/mmpose.git
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   pip3 install openmim
   mim install mmcv-full==1.3.9
   cd mmpose
   pip3 install -e .
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）


   获取[COCO](https://cocodataset.org/#download)数据集：coco2017，下载其中val2017图片及其标注文件，放入mmpose/data/coco/路径下，val2017目录存放coco数据集的验证集图片，“annotations”目录存放coco数据集的“person_keypoints_val2017.json”。目录结构如下：
   ```
   data
    ├──coco
         ├── annotations    //验证集标注信息文件夹      
         └── val2017             // 验证集文件夹
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   本操作封装在数据推理过程中，无须操作。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件，下载到mmpose源码仓目录下。
   
      ```
      wget https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth
      ```
      
   2. 导出onnx文件。
      1.使用ModelZoo获取的源码包中的文件替换mmpose目录下的文件
         ```
         cp pytorch2onnx.py mmpose/tools/deployment
         cp test.py mmpose/tools
         cp associative_embedding.py mmpose/mmpose/models/detectors
         ```

      2.使用pth导出ONNX。

         运行官方脚本导出ONNX。

         ```
         cd mmpose
         python3.7 tools/deployment/pytorch2onnx.py configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py ./hrnet_w32_coco_512x512-bcb8c247_20200816.pth --output-file ./hrnet.onnx
         ```
         - 参数说明：

           -   第一个参数表示使用的配置文件。
           -   第二个参数表示权重文件路径。
           -   第三个参数表示输出ONNX。

         当前目录获得hrnet.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/......
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
         bash atc_hrnet.sh Ascend${chip_name} # Ascend310P3
         示例
         bash atc_hrnet.sh Ascend310P3
         ```

         - 参数说明：

           -   第一个参数代表芯片类型。

           运行成功后生成hrnet.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        export PYTHONPATH='./mmpose/'
        python3 tools/test.py configs/body/2d_kpt_sview_rgb_img/assoiciate_embedding/coco/hrnet_w32_coco_512x512.py --out ./result.json --eval mAP
        ```

        -   参数说明：

             -   第一个参数表示配置文件路径
             -   第二个参数表示输出保存路径。
             -   第三个参数表示使用的评价指标。

    可以获得AR/AP数据，结果保存在result.json中。

   3.性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同档位的om模型的性能，参考命令如下：

        ```
        python3 -m ais_bench --model hrnet.om --dymDims input:1,3,{h},{w} --output ./ --outfmt BIN --loop 1000
        示例
        python3 -m ais_bench --model hrnet.om --dymDims input:1,3,512,512 --output ./ --outfmt BIN --loop 1000
        ```

      - 参数说明：
        - --model：需要验证om模型所在路径
        - --dymDims:需要验证模型的输入
        - --output：验证输出的保存位置
        - --outfmt：验证输出的保存格式
        - --loop:验证循环次数
      
      或者使用脚本map_postprocess.py进行统一验证，结果输出在屏幕上

       ```
       python3 map_postprocess.py
       ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集  | 精度                | 性能  |
| --------- |------------|------|-------------------|-----|
|   310P3        | 1          | coco | 0.653/AP 0.709/AR | 151 |

说明：源代码数据处理只支持单batch，OM为分档模型，性能是平均FPS性能