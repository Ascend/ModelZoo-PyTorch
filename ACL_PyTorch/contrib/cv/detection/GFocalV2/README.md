# GFocalV2模型-推理指导


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

GFocalV2主要引入边界框不确定性的统计量来高效地指导定位质量估计，从而提升one-stage的检测器性能。


- 参考实现：

  ```
  url=https://github.com/implus/GFocalV2
  commit_id=bfcc2b9fbbcad714cff59dacc8fb1111ce381cda
  ```




## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 800 x 1216 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | 100 x 5 | ND           |
  | output2  | FLOAT32  | 100 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动
  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/implus/GFocalV2.git
   cd GFocalV2         
   git checkout master    
   git reset --hard b7b355631daaf776e097a6e137501aa27ff7e757 
   patch -p1 < ../GFocalV2.diff
   python3 setup.py develop
   cd ..             
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用COCO数据集[coco2017](https://cocodataset.org/#download)，下载其中val2017图片及其标注文件，放入服务器/root/dataset/coco/文件夹，val2017目录存放coco数据集的验证集图片，annotations目录存放coco数据集的instances_val2017.json。目录结构如下：

   ```
   dataset
   ├── coco  
      ├── annotations  
      ├── val2017  
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行gfocal_preprocess.py脚本，完成预处理。

   ```
   python3 gfocal_preprocess.py --image_src_path=${datasets_path}/coco/val2017 --bin_file_path=val2017_bin --model_input_height=800 --model_input_width=1216
   ```

3. 生成预处理数据集信息文件

   执行get_info.py，生成数据集信息文件

   ```
   python3 get_info.py jpg ${datasets_path}/coco/val2017 gfocal_jpeg.info
   ```
   第一个参数为模型输入的类型，第二个参数为数据集路径，第三个为输出的info文件

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [gfocalv2预训练的pth权重文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/GFocalV2/PTH/gfocal_r50_fpn_1x.pth)

   2. 导出onnx文件。

      1. 使用pytorch2onnx.py导出onnx文件。

         ```
         python3 ./GFocalV2/tools/pytorch2onnx.py ./GFocalV2/configs/gfocal/gfocal_r50_fpn_1x.py ./gfocal_r50_fpn_1x.pth --output-file gfocal.onnx --input-img ./GFocalV2/demo/demo.jpg --shape 800 1216 --show
         ```

         获得gfocal.onnx文件。

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
         atc --framework=5 --model=./gfocal.onnx --output=gfocal_bs1 --input_format=NCHW --input_shape="input.1:1,3,800,1216" --log=debug --soc_version=Ascend${chip_name} 
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>***gfocal_bs1.om***</u>模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请点击本链接进行安装ais_bench推理工具，以及查看具体使用方法(https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)  

   2. 配置环境变量

        ```
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        ```

   3. 执行推理。


        ```
        python3 -m ais_bench --model ./gfocal_bs1.om --input ./val2017_bin --output result --outfmt TXT  
        ```

        -   参数说明：

             -   --model：om文件路径
             -   --input：预处理后二进制目录。
             -   --output：推理结果输出路径。
             -   --outfmt：推理结果输出格式。

        推理后的输出默认在当前目录result下。

        >**说明：** 
        >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见。

   4. 精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

      ```
      python3 gfocal_postprocess.py --bin_data_path=result/${outout_dir} --annotations_path=${datasets_path} --test_annotation=gfocal_jpeg.info --net_out_num=2 --net_input_height=800 --net_input_width=1216
      ```

      - 参数说明：

        - --bin_data_path：为生成推理结果所在目录
        - --annotations_path：为数据集annotation所在目录
        - --test_annotations：为预处理数据集信息文件所在路径

   5. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
        python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：om模型路径
        - --batchsize：batch大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|   310P        |      1            |    COCO2017        |   AP(IoU=0.50:0.95)=0.406         |      38.33 fps           |
|   310P        |      4            |    COCO2017        |   AP(IoU=0.50:0.95)=0.406         |      41.85 fps           |
|   310P        |      8            |    COCO2017        |   AP(IoU=0.50:0.95)=0.406         |      38.56 fps           |
|   310P        |      16            |    COCO2017        |   AP(IoU=0.50:0.95)=0.406         |      37.22 fps           |
|   310P        |      32            |    COCO2017        |   AP(IoU=0.50:0.95)=0.406         |      41.57 fps           |
|   310P        |      64            |    COCO2017        |   AP(IoU=0.50:0.95)=0.406         |      37.45 fps           |