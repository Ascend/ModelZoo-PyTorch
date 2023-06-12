# BLIP模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)





# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

BLIP模型为一种新的Vision-Language Pre-training框架，它可以灵活地转换到视觉语言理解和生成任务。Blip模型提出了一种用于合成网络图片字幕的字幕器，以及一种用于去除图像-文本对噪声的过滤器模型。本文档详细讲解使用BLIP模型进行图文检索任务推理的步骤。


- 参考实现：

  ```
	url=https://github.com/salesforce/BLIP
	branch=main
   commit_id=3a29b7410476bf5f2ba0955827390eb6ea1f4f9d
	model_name=BLIP
  ```
  
## 输入输出数据<a name="section540883920406"></a>

图文检索任务的推理步骤需分别提取文本与图像的多种嵌入特征，然后进行计算相似度，重排序等步骤，才可得到检索结果。由于该过程较为复杂，无法将所有步骤聚合在同一个模型中实现，因此将原始的BLIP检索模型拆分为三个模型，结合一系列后处理步骤，完成检索任务。

原始的BLIP检索模型拆分为以下三个模型：
- BLIP_text: 提取文本的低维嵌入表示，shape为256
- BLIP_image: 提取图像的低维嵌入表示，shape为256
- BLIP_image_feat：提取文本的高维特征表示，shape为577x768

#### BLIP_text

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | text_ids | INT64    | batchsize x 35          | ND          |
  | text_atten_mask | INT64    | batchsize x 35          | ND          |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | text_embed   | FLOAT32  | batchsize x 256 | ND           |

#### BLIP_image

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input_image    | RGB_FP32 | batchsize x 3 x 384 x 384 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | image_embed  | FLOAT32  | batchsize x 256 | ND           |

#### BLIP_image_feat

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input_image    | RGB_FP32 | batchsize x 3 x 384 x 384 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | image_feat   | FLOAT32  | batchsize x 577 x 768 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.7.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/salesforce/BLIP.git
   cd BLIP
   git reset --hard 3a29b7410476bf5f2ba0955827390eb6ea1f4f9d
   cd ..
   cp -r BLIP/models ./
   cp -r BLIP/configs ./
   cp BLIP/utils.py ./
   ```

2. 安装依赖。

   ```
   pip3 install -r requirement.txt
   ```



## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   COCO ： COCO数据集是一个大型的、丰富的物体检测，分割和字幕数据集。这个数据集以scene understanding为目标，主要从复杂的日常场景中截取，图像中的目标通过精确的segmentation进行位置的标定。图像包括91类目标，328,000影像和2,500,000个label。目前为止有语义分割的最大数据集，提供的类别有80 类，有超过33 万张图片，其中20 万张有标注，整个数据集中个体的数目超过150 万个。数据集下载地址：[http://cocodataset.org](http://cocodataset.org)。

	推理阶段仅需要`2014 val images`部分,包括约4万张图片，通过以下命令创建`coco2014`路径，下载数据集，解压并获取图像文件。
   ```
   mkdir -p ./coco2014
   wget http://images.cocodataset.org/zips/val2014.zip
   unzip -d ./coco2014 val2014.zip
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   下载测试集到`annotation`目录下。

   ```
   wget -P ./annotation https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json
   ```

   执行BLIP_preprocess.py脚本，完成预处理。

   ```
    python BLIP_preprocess.py \
        --coco_path ./coco2014 \
        --save_bin_path ./coco2014_bin 
   ```
   - 参数说明：
       -   --coco_path: 原始coco数据集所在路径
       -   --save_bin_path: 预处理后的二进制文件保存路径

   运行后在当前目录下的`save_bin_path`参数指定的路径中保存生成的二进制数据。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       训练权重链接为：https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth。
       在`BLIP`工作目录下可通过以下命令获取训练权重并转为推理模型。

         
      ```
      wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth
      ```

   2. 导出onnx文件。


      在`BLIP`工作目录下通过运行以下命令获取onnx模型。

      ```
      python BLIP_pth2onnx.py --pth_path ./model_base_retrieval_coco.pth
      ```
      
       - 参数说明：

            -   --pth_path：模型的pytorch预训练权重文件。

      运行后在当前工作目录下获得 `BLIP_text.onnx`，`BLIP_image.onnx`，`BLIP_image_feat.onnx` 三个文件。


   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/......
         ```

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3
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
            atc --framework=5 \
               --model=BLIP_image.onnx \
               --output=BLIP_image_bs${batchsize} \
               --input_format=NCHW \
               --input_shape="input_image:${batchsize},3,384,384" \
               --log=error \
               --soc_version=Ascend${chip_name}

            atc --framework=5 \
               --model=BLIP_image_feat.onnx \
               --output=BLIP_image_feat_bs${batchsize} \
               --input_format=NCHW \
               --input_shape="input_image:${batchsize},3,384,384" \
               --log=error \
               --soc_version=Ascend${chip_name}

            atc --framework=5 \
               --model=BLIP_text.onnx \
               --output=BLIP_text_bs${batchsize} \
               --input_format=ND \
               --input_shape="text_ids:${batchsize},35;text_atten_mask:${batchsize},35" \
               --log=error \
               --soc_version=Ascend${chip_name}
               --op_precision_mode=op_precision.ini
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。


           运行成功后生成`BLIP_image_bs${batchsize}.om`, `BLIP_image_feat_bs${batchsize}.om`, `BLIP_text_bs${batchsize}.om` 模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```
         mkdir coco2014_infer
         mkdir coco2014_infer/image_embed
         mkdir coco2014_infer/image_feat
         mkdir coco2014_infer/text_embed

         python -m ais_bench \
               --model ./BLIP_image_bs${batchsize}.om \
               --input ./coco2014_bin/img/ \
               --output ./coco2014_infer/image_embed  
         
         python -m ais_bench \
               --model ./BLIP_image_feat_bs${batchsize}.om \
               --input ./coco2014_bin/img/ \
               --output ./coco2014_infer/image_feat

         python -m ais_bench \
               --model ./BLIP_text_bs${batchsize}.om \
               --input ./coco2014_bin/ids/,./coco2014_bin/mask/ \
               --output ./coco2014_infer/text_embed  
        ```

        -   参数说明：

             -   model：om文件路径。
             -   input：输入数据路径
             -   output：输出数据路径。


   3. 精度验证。

      执行后处理脚本BLIP_postprocess.py`，参考命令如下：

      ```
      python BLIP_postprocess.py \
        --text_embed_path ./coco2014_infer/text_embed/${specific_dir} \
        --image_embed_path ./coco2014_infer/image_embed/${specific_dir} \
        --image_feat_path  ./coco2014_infer/image_feat/${specific_dir} \
        --coco_bin_path  ./coco2014_bin \
        --pth_path  ./model_base_retrieval_coco.pth 
      ```
      参数说明：
         - --text_embed_path  ais_bench推理输出的保存路径（文本嵌入）
         - --image_embed_path ais_bench推理输出的保存路径（图像嵌入）
         - --image_feat_path  ais_bench推理输出的保存路径（图像高维特征）
         - --coco_bin_path  预处理后的coco数据集的二进制文件保存路径
         - --pth_path  模型的pytorch预训练权重文件
      
      注意：ais_bench 工具在每次推理时会在指定的输出目录中新建一个以当前时间命名的文件夹作为最终的输出目录，因此在本步骤中需根据实际情况替换参数中的${specific_dir}。


   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python -m ais_bench --model=${om_model_path} --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：模型路径
        - --batchsize：每批次样本数量



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| 310P3          | 1                 | coco           | 81.3%           | text:292    |
| 310P3          | 4                | coco           |            |  text:754   |
| 310P3          | 8                | coco           |            |  text:1227   |
| 310P3          | 16                | coco           |            |  text:1400  |
| 310P3          | 32                | coco           |            |  text:1536   |
| 310P3          | 64                | coco           |            |   text:1662  |
| 310P3          | 1                 | coco           | 81.3%           | image:72    |
| 310P3          | 4                | coco           |            |  image:59  |
| 310P3          | 8                | coco           |            |  image:71  |
| 310P3          | 16                | coco           |            |  image:59  |
| 310P3          | 32                | coco           |            |  image:59  |
| 310P3          | 64                | coco           |            |  image:67  |
| 310P3          | 1                 | coco           | 81.3%           | image_feat: 73 |
| 310P3          | 4                | coco           |            |  image_feat: 59  |
| 310P3          | 8                | coco           |            |  image_feat:  70 |
| 310P3          | 16                | coco           |            |  image_feat: 59  |
| 310P3          | 32                | coco           |            |  image_feat: 58  |
| 310P3          | 64                | coco           |            |   image_feat: 67 |
