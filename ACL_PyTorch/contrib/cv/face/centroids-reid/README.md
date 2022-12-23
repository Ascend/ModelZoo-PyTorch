# Centroids-reid模型-推理指导


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

Centroids-reid是一种图像检索任务包括从一组图库（数据库）图像中找到与查询图像相似的图像。这样的系统用于各种应用，例如行人重新识别(ReID)或视觉产品搜索。




- 参考实现：

  ```
  url=https://github.com/mikwieczorek/centroids-reid
  commit_id=a1825b7a92b2a8d5e223708c7c43ab58a46efbcf
  model_name=centroids-reid
  ```
  





## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 256 x 128 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 2048 | ND           |




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
   git clone https://github.com/mikwieczorek/centroids-reid.git  
   cd ./centroids-reid
   git reset --hard a1825b7a92b2a8d5e223708c7c43ab58a46efbcf 
   patch -p1 <  centroid-reid.patch
   mkdir models
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）


   本模型支持[DukeMTMC-reID数据集](https://pan.baidu.com/share/init?surl=Oj78IrCnG_QSfhi9UY5mvA)，提取码为：eufe。下载后放在当前主目录下，目录结构如下：

   ```
   DukeMTMC-reID
   ├── bounding_box_test      
   └── query             
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行centroid-reid_preprocess.py脚本，完成预处理。

   ```
   mkdir -p DukeMTMC-reID/bin_data/gallery
   mkdir -p DukeMTMC-reID/bin_data/query
   python ./centroid-reid_preprocess.py  \
          --src_path DukeMTMC-reID/bounding_box_test  \
          --save_path DukeMTMC-reID/bin_data/gallery

   python ./centroid-reid_preprocess.py  \
          --src_path DukeMTMC-reID/query \
          --save_path DukeMTMC-reID/bin_data/query   
   ```
   - 参数说明：
      -   --src_path：数据集地址。
      -   --save_path：预处理结果保存在相应的文件夹。
 



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
      
      在该目录下获取权重文件

   2. 导出onnx文件。

      1. 使用centroid-reid_pth2onnx.py导出onnx文件。

         运行centroid-reid_pth2onnx.py脚本。

         ```
         python ./centroid-reid_pth2onnx.py  \
                --input_file ./models/dukemtmcreid_resnet50_256_128_epoch_120.ckpt \
                --output_file centroid-reid_r50_bs${bs}.onnx  \
                --batch_size ${bs}          
         ```

         获得centroid-reid_r50_bs${bs}.onnx文件。


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
         atc --framework=5 \
             --model=centroid-reid_r50_bs{bs}.onnx \
             --output=centroid-reid_r50_bs${bs} \
             --input_shape="input:${bs},3,256,128" \
             --input_format=NCHW \
             --soc_version=${chip_name}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --soc\_version：处理器型号。

           运行成功后生成<u>centroid-reid_r50_bs${bs}.om</u>模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请点击本链接进行安装ais_bench推理工具，以及查看具体使用方法(https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)  

   2. 执行推理。

        ```
        mkdir -p DukeMTMC-reID/result/gallery
        mkdir -p DukeMTMC-reID/result/query
        #gallery 
        python -m ais_bench \
                --model ./centroid-reid_r50_bs${bs}.om \
                --input ./DukeMTMC-reID/bin_data/gallery  \
                --output ./DukeMTMC-reID/result/gallery  \
                --outfmt TXT
                --batchsize ${bs}
        #query 
        python -m ais_bench \
				--model ./centroid-reid_r50_bs${bs}.om  \
				--input ./DukeMTMC-reID/bin_data/query  \
				--output ./DukeMTMC-reID/result/query  \
				--outfmt TXT  
				--batchsize ${bs}
        ```

        -   参数说明：

             -   model：om文件路径。
             -   input：预处理数据
             -   output：推理结果保存路径
             -   outfmt：推理输出类型



        >**说明：** 
        >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]。

   3. 精度验证。

      调用脚本centroid-reid_postprocess.py计算精度

      ```
      python ./centroid-reid_postprocess.py \
			   --dataset_dir ./DukeMTMC-reID/result/ \
			   --query_path query/${time_line}  \
			   --gallery_path  gallery/${time_line}
      ```

      - 参数说明：

        - dataset_dir：推理结果主目录

        - query_path：推理结果query目录

        - gallery_path：推理结果gallery目录

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7 -m ais_bench --model=${om_model_path} --loop=100 --batchsize=${batch_size}
        ```

      - 参数说明：
        - --model：模型地址
        - --batchsize：batchsize大小



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
|  Ascend310P3         |     1             |    DukeMTMC-reID        |    96.8%        |       1542          |
|  Ascend310P3         |     4             |    DukeMTMC-reID        |    96.8%        |       3715         |
|  Ascend310P3         |     8             |    DukeMTMC-reID        |    96.8%        |       4287          |
|  Ascend310P3         |     16             |    DukeMTMC-reID        |    96.8%        |       3454          |
|  Ascend310P3         |     32             |    DukeMTMC-reID        |    96.8%        |       3571          |
|  Ascend310P3         |     64             |    DukeMTMC-reID        |    96.8%        |      2231           |