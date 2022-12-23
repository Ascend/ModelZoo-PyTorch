#  ReID模型-推理指导


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

ReID是一个用于行人重识别任务的简单高效的Baseline模型，用于图像分类任务。


- 参考实现：

  ```
  url=https://github.com/michuanhaohao/reid-strong-baseline.git
  commit_id=3da7e6f03164a92e696cb6da059b1cd771b0346d
  code_path=ACL_PyTorch/contrib/cv/classfication/
  model_name=ReID
  ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 256 x 128 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | 1 x 1000 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17（NPU驱动固件版本为6.0.RC1） | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

- 该模型需要以下依赖   <u>***请不要将依赖列表与版本配套表合并***</u>

  **表 2**  依赖列表

  | 依赖名称       | 版本     |
  | -------------- | -------- |
  | onnx           | 1.7.0    |
  | Torch          | 1.8.0    |
  | TorchVision    | 0.9.0    |
  | torchaudio     | 0.8.0    |
  | numpy          | 1.20.3   |
  | Pillow         | 8.2.0    |
  | opencv-python  | 4.5.2.54 |
  | yacs           | 0.1.8    |
  | pytorch-ignite | 0.4.5    |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

    ```
   git clone https://github.com/michuanhaohao/reid-strong-baseline
   cd reid-strong-baseline 
   git reset 3da7e6f03164a92e696cb6da059b1cd771b0346d --hard
   cd ..
   ```
   
2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持  [Market1501数据集](https://gitee.com/link?target=http%3A%2F%2Fwww.liangzheng.org%2FProject%2Fproject_reid.html) 数据集。请用户需自行获取Market1501数据集，上传数据集到服务器任意目录并解压（如：/opt/npu/）。目录结构如下：

   ```
   ├── Market1501
     ├── bounding_box_test
     ├── query
     ...
   ```

2. 数据预处理。

   将原始数据集转换为模型输入的数据。

   将原始数据（.jpeg）转化为二进制文件（.bin）。

   执行两次预处理脚本ReID_preprocess.py，分别生成数据集query和数据集gallery预处理后的bin文件。
   
   ```
   python3 ReID_preprocess.py /home/HwHiAiUser/datasets/market1501/query prep_dataset_query
   python3 ReID_preprocess.py /home/HwHiAiUser/datasets/market1501/bounding_box_test prep_dataset_gallery
   mv prep_dataset_gallery/* prep_dataset_query/
   ```

   + 参数说明：
     + 第一个参数：原始数据验证集（.jpeg）所在路径。
     + 第二个参数：输出的二进制文件（.bin）所在路径。
   
   每个图像对应生成一个二进制文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

        下载pth权重文件   [pth权重文件](https://gitee.com/link?target=https%3A%2F%2Fdrive.google.com%2Fopen%3Fid%3D1hn0sXLZ5yJcxtmuY-ItQfYD7hBtHwt7A) 

        [网盘pth权重文件，提取码：v5uh](https://gitee.com/link?target=https%3A%2F%2Fpan.baidu.com%2Fs%2F1ohWunZOrOGMq8T7on85-5w) 
        
       文件名：market_resnet50_model_120_rank1_945.pth

       md5sum：0811054928b8aa70b6ea64e71ef99aaf 

   2. 导出onnx文件。

      1. 使用 ReID_pth2onnx.py导出onnx文件。

         运行pth2onnx脚本,
   
         ```
         python3 ReID_pth2onnx.py --config_file='reid-strong-baseline/configs/softmax_triplet_with_center.yml' MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('market_resnet50_model_120_rank1_945.pth')" 
         ```

         获得ReID.onnx文件。
   
         >**模型转换要点：** 
         
         >加上TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('no')"导出的onnx可以推理测试性能
            
         >不加上TEST.NECK_FEAT "('before')" TEST.FEAT_NORM "('no')"导出的onnx转换的om精度与官网精度一致 
   
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
         atc --framework=5 --model=ReID.onnx --output=ReID_bs1 --input_format=NCHW --input_shape="image:1,3,256,128" --log=debug --soc_version=Ascend${chip_name}
         ```
         
         - 参数说明：
         
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           
         
         运行成功后生成ReID_bs1.om模型文件。
   
2. 开始推理验证。

   1. 使用ais_bench工具进行推理。

      ais_bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。

        ```
        python3 -m ais_bench --model ./ReID_bs1.om --input ./prep_dataset_query/ --output ./ --output_dirname bs1 --outfmt BIN --batchsize 1
        ```

        -   参数说明：

             -   --model：om模型。
             -   --input：预处理数据集路径。
             -   --output：推理结果所在路径。
             -   --outfmt：推理结果文件格式。
             -   --output_dirname： 推理结果输出子文件夹。可选参数。与参数output搭配使用，单独使用无效。设置该值时输出结果将保存到 output/output_dirname文件夹中 。
             -   --batchsize：不同的batchsize。
   
        推理后的输出默认在当前目录result下。
   
        >**说明：** 
        >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见。
   
   3. 精度验证。
   
      调用ReID_postprocess.py脚本测试精度。
   
      ```
      python3 ReID_postprocess.py --query_dir=/root/datasets/market1501/query --gallery_dir=/root/datasets/market1501/bounding_box_test --pred_dir=./bs1
      ```
      
      - 参数说明：
      
        - --query_dir ： query数据集输入  
        - --gallery_dir： gallery数据集输入 
        - --pred_dir ：  ais_bench推理输出结果目录 

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

精度对比：

| Model      | ReID                      |
| ---------- | ------------------------- |
| 源码仓精度 | mAP：85.9%  Rank-1：94.5% |
| 310P3精度   | mAP：85.9%  Rank-1：94.5% |

性能：

| 芯片型号 | Batch Size   | 数据集 | 性能 |
| --------- | ---------------- | ---------- | --------------- |
| 310P3 | 1 | Market1501 | 1445.415 |
| 310P3 | 4 | Market1501 | 3558.945 |
| 310P3 | 8 | Market1501 | 4196.847 |
| 310P3 | 16 | Market1501 | 4417.628 |
| 310P3 | 32 | Market1501 | 2410.053 |
| 310P3 | 64 | Market1501 | 2234.776 |

