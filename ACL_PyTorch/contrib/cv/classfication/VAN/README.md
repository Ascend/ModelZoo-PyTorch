# VAN模型-推理指导

<!-- TOC -->

- [概述](#ZH-CN_TOPIC_0000001172161501)

  - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)
  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)
  
- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


<!-- /TOC -->

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

VAN模型基于一种新的大核注意（LKA）模块，以实现自注意中的自适应和远程关联，避免了在计算机视觉中图像的2D特性给自注意机制的应用所带来的问题。VAN虽然极其简单和高效，但在广泛的实验中，包括图像分类、对象检测、语义分割、实例分割等，其性能优于最先进的视觉变换器（ViTs）和卷积神经网络（CNN）。

- 参考实现：

  ```text
  url=https://github.com/Visual-Attention-Network/VAN-Classification
  branch=main
  commit_id=e19779b53a1b0828b51ecb4412d577541aee83a7
  model_name=Van
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | image    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型                  | 大小 | 数据排布格式 |
  | -------- | --------------------- | -------- | ------------ |
  | output   | FLOAT32 |  batchsize x 1000  | ND           |

# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.11.0  | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```shell
    git clone https://github.com/Visual-Attention-Network/VAN-Classification.git
    cd VAN-Classification
    git checkout main
    git reset --hard e19779b53a1b0828b51ecb4412d577541aee83a7
    cd ..
   ```

2. 安装依赖。

   ```shell
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集
   本模型使用[ImageNet官网](https://gitee.com/link?target=http%3A%2F%2Fwww.image-net.org)的5万张验证集进行测试，以ILSVRC2012为例，用户需获取[ILSVRC2012数据集](http://www.image-net.org/download-images)，并上传到服务器，图片与标签分别存放在/opt/npu/imagenet/val与/opt/npu/imageNet/val_label.txt。

   ```
   ├── imagenet
       ├── val
       ├── val_label.txt 
   ```
   
2. 数据预处理。

    运行preprocess.py脚本对数据进行预处理

    ```shell 
    python3 ./VAN_preprocess.py VAN ${scr_path}/val ./${save_path}
    ```

    参数说明：

    - model_type：数据预处理方式为VAN
    - ${scr_path}/val：原始数据验证集（.jpeg）所在路径
    - ${save_path}：输出的二进制文件（.bin）存放路径


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

    使用PyTorch将模型权重文件.pth.tar转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

    1. 获取 [权重文件](https://cloud.tsinghua.edu.cn/f/58e7acceaf334ecdba89/?dl=1)


    2. 导出onnx文件

       1. 复制van.py到当前目录。

           ```
           cp ./VAN-Classification/models/van.py ./
           ```

       2. 使用pth2onnx.py导出onnx文件，运行VAN_pth2onnx.py脚本。

           ```shell
           python3 VAN_pth2onnx.py ./${onnx_path} ./${van.pth}
           ```

           参数说明：

           - ${onnx_path}：onnx模型的保存路径
           - ${van_pth}：模型权重文件

    3. 使用ATC工具将ONNX模型转OM模型。

        1. 配置环境变量。

            ```shell
            source /usr/local/Ascend/ascend-toolkit/set_env.sh
            ```

            > **说明：**
            > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

        2. 执行命令查看芯片名称。

            ```shell
            npu-smi info
            
            回显如下：
            +-------------------+-----------------+------------------------------------------------------+
            | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
            | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
            +===================+=================+======================================================+
            | 0       310P3     | OK              | 16.8         53                0    / 0              |
            | 0       0         | 0000:86:00.0    | 0            944  / 21534                            |
            +===================+=================+======================================================+
            ```

        3. 执行ATC命令。

            ```shell
            atc --framework=5 \
                --model=${onnx_path} \
                --input_format=NCHW \
                --input_shape="image:16,3,224,224" \
                --output=${om_path} \
                --op_precision_mode=op_precision.ini \
                --soc_version=${chip_name}
            ```

            参数说明：
            - --framework：5代表ONNX模型。
            
            - --model：为ONNX模型文件。
            
            - --output：输出的OM模型。
            
            - --input\_format：输入数据的格式。
            
            - --input\_shape：输入数据的shape。
            
            - --soc\_version：处理器型号。
            
            - --op\_precision\_mode:  指定部分算子选择高性能模式。
            
                运行成功后生成om模型文件，推荐在模型名后加后缀，如‘_bs1’，用以区分不同batch_size的om模型。

2. 开始推理验证。

    1. 使用ais_bench工具进行推理。

        ais_bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

    2. 执行推理。

        ```shell
        python3 -m ais_bench --model ${om_path} --input ${prep_dataset} --output ${output} --outfmt TXT --batchsize=${bs}
        ```

        参数说明：
        - --model：om文件路径
        - --input：输入数据的路径
        - --output：推理结果存放路径
        - --outfmt：输出数据的格式
    
3. 精度验证。

    调用VAN_postprocess.py脚本将推理结果与label进行比对，结果保存在result.json

    ```shell
    python3 VAN_postprocess.py --anno_file=${val_label.txt} --benchmark_out=${result_path}} --result_file=./result.json
    ```

    - 参数说明：
         - --anno_file：val_label.txt数据标签存放路径
         - --benchmark_out：模型推理结果存放路径
         - --result_file：精度结果存放路径

4. 性能验证。

    可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

    ```shell
    python3-m ais_bench --model=${van_bs.om} --loop=100 --batchsize=${bs}
    ```

    - 参数说明：
      - --model：om模型
      - --batchsize：om模型的batchsize
度
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度数据

| 芯片型号 | Batch Size | 数据集    | 测试精度top1  | 标杆精度top1 |
| ------- | ---------- | -------- | ----- | ---- |
| 310P3   | 1          | imageNet2012 | 82.78% | 82.8% |
| 310P3   | 16          | imageNet2012 | 82.78% | 82.8% |


性能数据

| Batch Size | 数据集       | 310P(FPS/Card) | 
| ---------- | ------------ | -------------- |
| 1          | imageNet2012 | 379.953649     | 
| 4          | imageNet2012 | 764.789115     | 
| 8          | imageNet2012 | 874.938485     |
| 16         | imageNet2012 | 854.965747     | 
| 32         | imageNet2012 | 805.06184      |
| 64         | imageNet2012 | 内存不足       | 

