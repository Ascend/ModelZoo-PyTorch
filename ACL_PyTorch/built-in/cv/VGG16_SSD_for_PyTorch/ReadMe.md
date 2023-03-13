# VGG16_SSD 模型-推理指导


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

SSD网络是继YOLO之后的one-stage目标检测网络，是为了改善YOLO网络设置的anchor设计的太过于粗糙而提出的，其设计思想主要是多尺度多长宽比的密集锚点设计和特征金字塔。


- 参考实现：

  ```
  url=https://github.com/qfgaohao/pytorch-ssd.git
  commit_id=88c0311442b6dbbe2cacf06c5fcc3a68d85aa50c
  model_name=VGG16_SSD
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据       | 数据类型 | 大小                      | 数据排布格式 |
  | -------------- | -------- | ------------------------- | ------------ |
  | actual_input_1 | RGB_FP32 | batchsize x 3 x 300 x 300 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小                  | 数据排布格式 |
  | -------- | -------- | --------------------- | ------------ |
  | scores   | FLOAT32  | batchsize x 8732 x 21 | ND           |
  | boxes    | FLOAT32  | batchsize x 8732 x 4  | ND           |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/qfgaohao/pytorch-ssd.git
   ```
   
2. 安装依赖

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持VOC2007 4952张图片的验证集。请用户需自行获取[VOC2007数据集](http://host.robots.ox.ac.uk/pascal/VOC)，上传数据集到服务器任意目录并解压（如：/home/datasets/VOCdevkit/）。
   解压后数据集目录结构：
   ```
   └─VOCdevkit
       └─VOC2007
           ├──SegmentationObject # 实例分割图像
           ├──SegmentationClass  # 语义分割图像
           ├──JPEGImages         # 训练集和验证集图片
           ├──Annotations        # 图片标注信息（label）
           ├──ImageSets          # 训练集验证集相关数据
           │    ├── Segmentation
           │    ├── Main
           │    └── Layout
   ```
   
2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行ssd_pth_preprocess.py脚本，完成预处理
   ```
   python3 ssd_pth_preprocess.py vgg16_ssd ./VOC2007/JPEGImages/ ./prep_bin
   ```



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       到对应git仓获取VGG SSD对应的权重文件[vgg16-ssd-mp-0_7726.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/VGG16/PTH/vgg16-397923af.pth)
       
   2. 导出onnx文件。

      1. 使用vgg16_ssd_pth2onnx.py导出onnx文件

         运行vgg16_ssd_pth2onnx.py脚本。

         ```
         python3.7 vgg16_ssd_pth2onnx.py ./vgg16-ssd-mp-0_7726.pth vgg16_ssd.onnx
         ```
   
         获得`vgg16_ssd.onnx`文件。
      
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
            atc --model=./vgg16_ssd.onnx --framework=5 --output=vgg16_ssd_bs{batch_size} --input_format=NCHW --input_shape="actual_input_1:{batch_size},3,300,300" --log=info --soc_version={chip_name} 
            示例
            atc --model=./vgg16_ssd.onnx --framework=5 --output=vgg16_ssd_bs1 --input_format=NCHW --input_shape="actual_input_1:1,3,300,300" --log=info --soc_version=Ascend310P3
            ```
   
            - 参数说明：
   
              -   --model：为ONNX模型文件。
              -   --framework：5代表ONNX模型。
              -   --output：输出的OM模型。
              -   --input\_format：输入数据的格式。
              -   --input\_shape：输入数据的shape。
              -   --log：日志级别。
              -   --soc\_version：处理器型号。
            
   
              运行成功后生成`vgg16_ssd_bs{batch_size}.om`模型文件。
   
2. 开始推理验证

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
        python3 -m ais_bench --model vgg16_ssd_bs{batch_size}.om \
   				--input prep_bin \
   				--output ./ \
   				--output_dirname result \
   				--outfmt BIN
        示例
        python3 -m ais_bench --model vgg16_ssd_bs1.om \
   				--input prep_bin \
   				--output ./ \
   				--output_dirname result \
   				--outfmt BIN
        ```

        -   参数说明：

             -   model：om模型
             -   input：推理输入数据
             -   output：推理结果输出路径
             -   output_dirname: 推理结果文件夹
             -   outfmt: 推理结果格式
                  	

        推理后的输出默认在当前目录result下。

        >**说明：** 
        >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   3. 精度验证。

      调用ssd_pth_postprocess.py脚本推理结果与label比对，可以获得每个类别以及所有类别的精度，结果保存在eval_results中。
      ```
      python3 ssd_pth_postprocess.py ./VOC2007/ ./voc-model-labels.txt ./result/ ./eval_results/
      ```
      - 参数说明
        - ./result: 推理结果
        - voc-model-labels.txt: 验证label
        - ./eval_results/: 保存结果路径

      
4. 性能验证
      可使用ais_infer推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
   
   ```
   python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
   示例
   python3 -m ais_bench --model=vgg16_ssd_bs1.om --loop=20 --batchsize=1
   ```
   
   - 参数说明：
        - --model：om模型
        - --batchsize：模型batchsize
        - --loop: 循环次数



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度 | 性能 |
| -------- | ---------- | ------ | ---- | ---- |
|     310P3     |    1        | VOC2007 |   acc:0.7726   |  477  |
|     310P3     |    4        | VOC2007 | acc:0.7726 |  717  |
|     310P3     |    8        | VOC2007 | acc:0.7726 |  738  |
|     310P3     |    16        | VOC2007 | acc:0.7726 |  751  |
|     310P3     |    32        | VOC2007 |   acc:0.7726   |  730  |
|     310P3     |    64        | VOC2007 |   acc:0.7726   |  718  |

说明：精度是所有类别的平均值