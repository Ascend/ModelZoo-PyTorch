# YOLACT模型-推理指导


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

YOLACT是2019年发表在ICCV上面的一个实时实例分割的模型，它主要是通过两个并行的子网络来实现实例分割的。(1)Prediction Head分支生成各个anchor的类别置信度、位置回归参数以及mask的掩码系数；(2)Protonet分支生成一组原型mask。然后将原型mask和mask的掩码系数相乘，从而得到图片中每一个目标物体的mask。论文中还提出了一个新的NMS算法叫Fast-NMS，和传统的NMS算法相比只有轻微的精度损失，但是却大大提升了分割的速度。

- 参考论文：
   [Bolya D, Zhou C, Xiao F, et al. Yolact: Real-time instance segmentation[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 9157-9166.](https://arxiv.org/abs/1608.06993)


- 参考实现：

  ```
  url=https://github.com/dbolya/yolact.git
  branch=master 
  commit_id=57b8f2d95e62e2e649b382f516ab41f949b57239 
  model_name=YOLACT
  ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 550 x 550 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | :--------: | ------------ |
  | output1  | FLOAT32  | batchsize x 19248 x 32 | ND           |
  | output2  | FLOAT32  | batchsize x 19248 x 81 | ND           |
  | output3  | FLOAT32  | batchsize x 19248 x 4 | ND           |
  | output4  | FLOAT32  | batchsize x 138 x 138 x 32 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/dbolya/yolact
   cd yolact
   git checkout master
   git reset --hard 57b8f2d95e62e2e649b382f516ab41f949b57239

   patch -p1 < ../YOLACT.patch
   
   cp ../YOLACT_preprocess.py ./
   cp ../YOLACT_postprocess.py  ./
   cp ../YOLACT_pth2onnx.py ./
   ```

   目录结构如下：
   ```
   ├─YOLACT
      ├─yolact                              //开源仓位置
         ├─datasets                         //数据集位置
         ├─YOLACT_pth2onnx.py
         ├─YOLACT_preprocess.py
         ├─YOLACT_postprocess.py
      ├─modelzoo_level.txt
      ├─YOLACT.patch
      ├─requirements.txt
      ├─LICENSE
      ├─README.md
   ```

2. 安装依赖。

   ```
   pip3 install -r ../requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持[COCO2017 验证集](https://cocodataset.org/#home)共4952张图片。可上传val2017文件夹和instances_val2017.json文件到路径下，以"./datasets"为例。目录结构如下：

   ```
   ├──./datasets
      ├── instances_val2017.json    //验证集标注信息       
      └── val2017             // 验证集文件夹
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行YOLACT_preprocess.py脚本，完成预处理。

   ```
   python3 YOLACT_preprocess.py --valid_images=./datasets/val2017 --valid_annotations=./datasets/instances_val2017.json --saved_path=./prep_dataset
   ```

   - 参数说明：
      - --valid_images：验证集文件路径。
      - --valid_annotations：验证集标注信息路径。
      - --saved_path：预处理后的bin文件路径。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      下载权重文件[yolact_base_54_800000.pth](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EYRWxBEoKU9DiblrWx2M89MBGFkVVB_drlRd_v5sdT3Hgg)，可放置于任意路径下，以"./"目录下为例。
      

   2. 导出onnx文件。

      使用YOLACT_pth2onnx.py导出onnx文件。

         运行YOLACT_pth2onnx.py脚本。

         ```
         python3 YOLACT_pth2onnx.py --trained_model=yolact_base_54_800000.pth --outputName=./yolact --dynamic=True
         ```

         - 参数说明：
            - --trained_model：权重文件路径。
            - --outputName：onnx模型文件名的路径。
            - --dynamic：预处理后的bin文件路径。

         获得./yolact.onnx文件

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
         atc --framework=5 --model=yolact.onnx --output=yolact_bs1 --input_format=NCHW --input_shape="input.1:1,3,550,550" --log=error --soc_version=Ascend${chip_name} 
         ```

         - 参数说明：
           - --model：为ONNX模型文件。
           - --framework：5代表ONNX模型。
           - --output：输出的OM模型。
           - --input\_format：输入数据的格式。
           - --input\_shape：输入数据的shape。
           - --log：日志级别。
           - --soc\_version：处理器型号。

           运行成功后生成yolact_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理(${ais_infer_path}请根据实际的推理工具路径填写)。

      ```
      mkdir result
      python3 -m ais_bench --model=./yolact_bs1.om --input=./prep_dataset --output=./result --output_dirname=bs1 --outfmt=BIN --batchsize=1 --device=0
      ```

      - 参数说明：
         -  --model：om文件路径。
         -  --input：输入的bin文件路径。
         -  --output：推理结果文件路径。
         -  --outfmt：输出结果格式。
         -  --batchsize：批大小。
         -  --device：NPU设备编号。

        推理后的输出默认在当前目录./result/bs1下。


   3. 精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

      ```
      python3 YOLACT_postprocess.py --valid_images=./datasets/val2017 --valid_annotations=./datasets/instances_val2017.json --npu_result=./result/bs1
      ```

      - 参数说明：

        - --valid_images：验证集文件路径。
        - --valid_annotations：验证集标注信息路径。
        - --npu_result：om模型的推理结果路径。

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size} --device=${device_id}
      ```

      - 参数说明：
         - --model：om模型路径。
         - --batchsize：批大小。
         - --loop：推理循环次数。
         - --device：NPU设备编号。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| :---------: | :-----:| :----------: | :----------: | :----------: |
| Ascend310P |   1   |   coco2017   |  box: 32.07, mask: 29.72  | 163.439 |
| Ascend310P |   4   |   coco2017   |  box: 32.07, mask: 29.72  | 131.034 |
| Ascend310P |   8   |   coco2017   |  box: 32.07, mask: 29.72  | 113.595 |
| Ascend310P |   16   |   coco2017   |  box: 32.07, mask: 29.72  | 117.710 |
| Ascend310P |   32   |   coco2017   |  box: 32.07, mask: 29.72  | 120.386 |
| Ascend310P |   64   |   coco2017   |  box: 32.07, mask: 29.72  | 118.647 |