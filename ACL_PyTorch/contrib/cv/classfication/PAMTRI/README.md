# PAMTRI模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

   - [输入输出数据](#ZH-CN_TOPIC_0000001126281702)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

PAMTRI是一种姿态感知多任务重新识别框架，它通过关键点、热图和来自姿态估计的片段明确推理车辆姿态和形状，从而克服了视点依赖性，并在执行 ReID 时联合对语义车辆属性（颜色和类型）进行分类，通过具有嵌入式姿势表示的多任务学习。本文档描述的PAMTRI是基于PyTorch实现的版本。

- 参考实现：

  ```
  url=https://github.com/NVlabs/PAMTRI
  commit_id=a835c8cedce4ada1bc9580754245183d9f4aaa17
  code_path=PAMTRI/MultiTaskNet
  model_name=MultiTaskNet
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 256 x 256 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | 1 x 1024 | ND           |
  | output2  | FLOAT32  | 1 x 10   | ND           |
  | output3  | FLOAT32  | 1 x 9    | ND           |
  | output4  | FLOAT32  | 1 x 575  | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                            | 版本    | 环境准备指导                                                                                          |
  | --------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
  | 固件与驱动                                                      | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                            | 5.1.RC2 | -                                                                                                     |
  | Python                                                          | 3.7.13  | -                                                                                                     |
  | PyTorch                                                         | 1.9.0   | -                                                                                                     |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码

1. 获取源码。

   在已下载的源码包根目录下，执行如下命令。

   ```shell
   git clone https://github.com/NVlabs/PAMTRI
   cd PAMTRI/MultiTaskNet
   git checkout master
   git reset --hard 25564bbebd3ccf11d853a345522e2d8c221b275d
   patch -p1 < ../../densenet.patch
   cd -
   ```

2. 安装依赖。

   ```shell
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型使用veri数据集，用户需自行获取数据集，参考  [PROVID Progressive and Multi-modal Vehicle Re-identification for Large-scale Urban Surveillance (vehiclereid.github.io)](https://vehiclereid.github.io/VeRi/) 获取数据集，将获取到的VeRi数据集内容解压至 `MultiTaskNet/data` 文件夹内。目录结构如下：

   ```
   data
    ├──veri                    //veri数据集
       ├── image_train         //VeRi数据集训练数据，本推理不需使用
       └── image_query         //VeRi数据集验证数据
       └── image_test          //VeRi数据集测试数据
       └── label_train.csv     //VeRi数据集训练数据的标签数据
       └── label_query.csv     //VeRi数据集验证数据的标签数据
       └── label_test.csv      //VeRi数据集测试数据的标签数据
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行PAMTRI_preprocess.py脚本，完成预处理。

   ```shell
   python PAMTRI_preprocess.py \
	 			  --query_dir PAMTRI/MultiTaskNet/data/veri/image_query \
              --gallery_dir PAMTRI/MultiTaskNet/data/veri/image_test \
              --save_query ./prep_dataset_query \
              --save_gallery ./prep_dataset_gallery
   ```

   - 参数说明
     - --save_query：输出query的二进制文件（.bin）所在路径。
     - --save_gallery：输出gallery的二进制文件（.bin）所在路径。
     - --query_dir：query数据集的路径。
     - --gallery_dir：gallery数据集的路径。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [权重文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/PAMTRI/PTH/densenet121-a639ec97.pth)都存放在源码库的models目录下，可直接使用

   2. 导出onnx文件。

      使用PAMTRI_pth2onnx.py导出onnx文件。

      ```shell
      python PAMTRI_pth2onnx.py \
                --load-weights models/densenet121-xent-htri-veri-multitask/model_best.pth.tar \
                --output_path ./PAMTRI_dynamic.onnx \
                --multitask
      ```

      获得PAMTRI_dynamic.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

       1. 配置环境变量。

          ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
          ```

       2. 执行命令查看芯片名称（$\{chip\_name\}）。

          ```shell
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

         将下命令中的${chip_name}改为当前处理器型号, {batchsize}改为实际输入的batchsize

         ```shell
         batch_size=1               # 本文仅以batch_size=1进行说明
         atc --framework=5 \
              --model=PAMTRI_dynamic.onnx \
              --output=PAMTRI_bs${batch_size} \
              --input_format=NCHW \
              --input_shape="input:${batch_size},3,256,256" \
              --log=debug \
              --soc_version=Ascend${chip_name}
         ```

         - 参数说明：
           - --model：为ONNX模型文件。
           - --framework：5代表ONNX模型。
           - --output：输出的OM模型。
           - --input\_format：输入数据的格式。
           - --input\_shape：输入数据的shape。
           - --log：日志级别。
           - --soc\_version：处理器型号。
           - --insert\_op\_conf=aipp\_resnet34.config:  AIPP配置

         运行成功后生成<u>***PAMTRI_bs1.om***</u>模型文件。


2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。
      > `output` 路径根据用户需求自由设置，这里以 `output=./out` 为例说明
      ```shell
      # 针对query数据集进行推理
      python -m ais_bench \
         --model "./PAMTRI_bs1.om" \
         --input "./prep_dataset_query" \
         --output ${output} \
         --output_dirname dataset_query_features \
         --outfmt BIN \
         --device 0 \
         --batchsize 1 \
         --loop 1

      # 针对gallery数据集进行推理
      python -m ais_bench \
         --model "./PAMTRI_bs1.om" \
         --input "./prep_dataset_gallery" \
         --output ${output} \
         --output_dirname dataset_gallery_features \
         --outfmt BIN \
         --device 0 \
         --batchsize 1 \
         --loop 1
      ```

      -   参数说明：

          -   --model：om文件路径。
          -   --input:输入路径
          -   --output：输出路径。
          -   --output_dirname：输出数据的子目录名称
          -   --outfmt：输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”。
          -   --loop：推理次数，可选参数，默认1，profiler为true时，推荐为1

         推理后的输出默认在 `--output` 文件夹下。


   3. 精度验证。

      执行后处理脚本文件PAMTRI_postprocess.py评测mAP精度，结果保存在result.json中。精度验证之前，将推理结果文件中summary.json删除。将{batchsize}改为实际的batchsize

      ```shell
      python PAMTRI_postprocess.py \
            --queryfeature_path=./result/dataset_query_features \
            --galleryfeature_path=./result/dataset_gallery_features \
            --test-batch ${batchsize} > result.json
      ```

      - 参数说明：

        - --queryfeature_path：生成推理结果所在路径。
        - --galleryfeature_path：生成推理结果所在路径。
        - result_bs1.json：生成结果文件。
        - --test-batch：推理数据的batchsize。

      执行完成后会在当前目录下生成result.json，保存结果的mAP精度值

   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```shell
        python -m ais_bench \
              --model ./PAMTRI_bs1.om \
              --input ./prep_dataset_query \
              --loop 20 \
              --batchsize 1
        ```

      - 参数说明：
        - --model：om文件路径。
        - --input ：输入的数据集路径
        - --batchsize：每次输入模型的样本数。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度（mAP） | 性能     |
| -------- | ---------- | ------ | ----------- | -------- |
| 310P3    | bs1        | veri   | 68.64%      | 678.067  |
| 310P3    | bs4        | veri   | 68.64%      | 1564.274 |
| 310P3    | bs8        | veri   | 68.64%      | 1540.995 |
| 310P3    | bs16       | veri   | 68.64%      | 1265.894 |
| 310P3    | bs32       | veri   | 68.64%      | 1265.894 |