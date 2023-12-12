# YOLOX_tiny 模型-推理指导


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

YOLOX对YOLO系列进行了一些有经验的改进，将YOLO检测器转换为无锚方式，并进行其他先进的检测技术，即解耦头和领先的标签分配策略SimOTA，在大规模的模型范围内获得最先进的结果。


- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection.git
  commit_id=3e2693151add9b5d6db99b944da020cba837266b
  model_name=YoloX
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32  | batchsize x 3 x 640 x 640 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                     | 数据排布格式 |
  | -------- | -------- | ------------------------ | ------------ |
  | dets     | FLOAT32  | batchsize x num_dets x 5 | ND           |
  | labels   | INT32    | batchsize x num_dets     | ND           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   git reset 3e2693151add9b5d6db99b944da020cba837266b --hard
   pip3 install -v -e .
   mmdetection_path=$(pwd)
   cd ..
   git clone https://github.com/open-mmlab/mmdeploy.git
   cd mmdeploy
   git reset 0cd44a6799ec168f885b4ef5b776fb135740487d --hard
   pip3 install -v -e .
   mmdeploy_path=$(pwd)
   cd ..
   ```
   
2. 安装依赖

   ```
   pip3 install -r requirements.txt
   pip3 install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index/html
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   1. 本模型支持coco2017验证集。用户需自行获取数据集，将annotations文件和val2017文件夹解压并上传数据集到源码包路径下的dataset文件夹下。目录结构如下：

      ```
      ├── annotations
      └── val2017 
      ```
2. 数据预处理
   1. 将原始数据转化为二进制文件（.bin）。
   执行YOLOX_preprocess.py脚本，生成数据集预处理后的bin文件，存放在当前目录下的val2017_bin文件夹中。
   ```
   python3 YOLOX_preprocess.py --image_src_path /root/datasets/coco/val2017 --bin_file_path ./val2017_bin --meta_file_path ./val2017_bin_meta
   ```   
   - 参数说明：
     - --image_src_path：为数据集路径。
     - --bin_file_path：二进制文件夹路径
     - --meta_file_path：保存预处理scalar的meta数据文件路径


   2. 获取二进制数据集信息
   ```
   python3 gen_dataset_info.py /root/datasets/coco ${mmdetection_path}/configs/yolox/yolox_tiny_8x8_300e_coco.py val2017_bin val2017_bin_meta yolox.info yolox_meta.info 640 640
   ```
   执行gen_dataset_info.py生成yolox.info和yolox_meta.info，命令最后的俩个640分别是图片的宽和高，生成的info文件会在后处理中用到

   - 参数说明：
     - /root/datasets/coco：为数据集路径(val2017的上层目录，即coco的路径)
     - ${mmdetection_path}/configs/yolox/yolox_s_8x8_300e_coco.py：固定参数，为该模型的配置
     - val2017_bin：二进制文件路径
     - val2017_bin_meta：二进制meta文件路径


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   1. [yolox_tiny](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth)下载YOLOX-s对应的weights， 名称为yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth。放到mmdeploy_path目录下

   2. 数据集运行转模型脚本。

   ```
   bash test/pth2onnx.sh ${mmdetection_path} ${mmdeploy_path}
   ```
   会在mmdeploy/work_dir中生成end2end.onnx模型文件

   3. 执行命令查看芯片名称（$\{chip\_name\}）。

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

   4. 执行ATC命令。

   ```
   #配置环境变量
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   atc --framework=5 --model=${mmdeploy_path}/work_dir/end2end.onnx --output=yolox_bs8 --input_format=NCHW --op_precision_mode=op_precision.ini --input_shape="input:8,3,640,640" --log=error --soc_version=Ascend${soc_version}
   ```
   - 参数说明：
     - --model：为ONNX模型文件。
     - --framework：5代表ONNX模型。
     - --output：输出的OM模型。
     - --input_format：输入数据的格式。
     - --input_shape：输入数据的shape。
     - --log：日志级别。
     - --soc_version：处理器型号。
   生成yolox_bs8.om模型文件
   
2. 开始推理验证

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装

   2. 执行推理。
      ```
      python3 -m ais_bench --model ./yolox_bs8.om --input val2017_bin --output ./outputs --outfmt BIN
      ```
      - 参数说明：
        - --model：om模型的路径
        - --input：二进制数据集路径
        - --output：保存推理结果的路径（未指定具体文件名时，会在该文件夹下生成时间戳文件夹保存推理结果）
        - --outfmt：输出文件格式，由于后处理脚本要求，此处为BIN

   3. 精度测试
      ```
      python3 YOLOX_postprocess.py --dataset_path /root/datasets/coco --model_config ${mmdetection_path}/configs/yolox/yolox_tiny_8x8_300e_coco.py --bin_data_path ./outputs/2023_03_06-15_16_37/
      ```
      - 参数说明：
        - --datasets_path：coco数据集的路径(路径中不需要加val2017)
        - --model_config：模型配置文件
        - --bin_data_path：保存二进制推理结果的文件夹路径，需替换成对应的文件夹，路径末尾的/需要加上
      执行精度测试后会在当前文件夹下生成results.txt文件，用于保存bbox_map的值


   4. 性能测试：

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model=./yolox_bs8.om --loop=10
      ```

      - 参数说明：
         - --model：om模型


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据，高于主仓精度

| 芯片型号 | Batch Size | 数据集 | 精度 | 310P性能 |
| -------- | ---------- | ------ | ---- | ---- |
|     310P3     |   1   | coco2017 | bbox_map:0.3336 |  303.55fps  |
|     310P3     |   4   | coco2017 | bbox_map:0.3336 |  684fps  |
|     310P3     |   8   | coco2017 | bbox_map:0.3336 |  682fps  |
|     310P3     |   16   | coco2017 | bbox_map:0.3336 |  636fps  |
|     310P3     |   32   | coco2017 | bbox_map:0.3336 |  617fps  |
|     310P3     |   64   | coco2017 | bbox_map:0.3336 |  621fps  |
