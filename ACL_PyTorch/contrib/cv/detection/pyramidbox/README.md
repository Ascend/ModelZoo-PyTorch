# Pyramidbox模型-推理指导


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

Pyramidbox是一种新的基于上下文辅助的单镜头人脸检测器。首先，通过使用一个新的上下文锚，通过一种半监督的方法来监督高级上下文特征的学习，；其次，提出了低层次特征pyramid网络，将足够的高层次上下文语义特征和低层次面部特征结合在一起，这也使得模型能够在单一镜头中预测所有尺度的人脸；第三，引入上下文敏感结构，增加预测网络的容量，提高最终输出的精度。此外，采用数据锚采样的方法对不同尺度的训练样本进行增广，增加了小人脸训练数据的多样性。


- 参考实现：

  ```
  url=https://gitee.com/kghhkhkljl/pyramidbox.git
  commit_id=b498eefe0bd7ce6a530c195642e836c314b57c81
  code_path=ACL_PyTorch/contrib/cv/detection
  model_name=pyramidbox
  ```
  

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- | -------- | --------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 1000 x 1000 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | output1  | FLOAT16  | batchsize x 1000 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.9.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```shell
   git clone https://gitee.com/kghhkhkljl/pyramidbox.git        # 克隆仓库的代码
   ```
   
   克隆下来源代码之后将pr中的代码放到克隆下来的pyramidbox下面
   
2. 安装依赖。

   ```shell
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持widerface 3226张图片的验证集。下载地址：[https://share.weiyun.com/5ot9Qv1](https://share.weiyun.com/5ot9Qv1)

   上传数据集到服务器pyramidbox目录并解压存放至images（需自建）。目录结构如下：

   ```
   images
   ├── 0--Parade    //不同二级目录       
   └── 1--Handshaking 
   └── ...
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行“pyramidbox_preprocess.py”脚本，完成预处理。

   ```shell
   cd pyramidbox
   python3 pyramidbox_preprocess.py ./images ./data1000_1 ./data1000_2
   ```

   * 参数说明
     * 第一个参数：数据集所在目录
     * 第二个和第三个参数：预处理后的文件名（说明：由于预处理需要进行两次图片的不同处理，所以生成的文件有两个）

3. 迁移数据集信息文件

   检索预处理完成后的结果并转移至新文件夹内

   ```shell
   find ${dataset_path}/data1000_1 -type f -iname "*.bin" -exec mv --backup=numbered -t ${dataset_path}/bs1_data_1 {} +
   find ${dataset_path}/data1000_2 -type f -iname "*.bin" -exec mv --backup=numbered -t ${dataset_path}/bs1_data_2 {} +
   
   需要先新建用以存放迁移后数据的文件夹，因为预处理结果额外包含一层子目录，所以将所有后缀符合的文件全部取出放入新文件夹内
   ```


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      ```shell
      wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/pyramidbox/PTH/pyramidbox_120000_99.02.pth
      ```

   2. 使用pth2onnx.py进行onnx的转换。

      ```shell
      python3 pyramidbox_pth2onnx.py  ./pyramidbox_1000.onnx ./pyramidbox_120000_99.02.pth
      ```

      * 参数说明：
        * 第一个参数：onnx文件生成在当前目录的名字。
        * 第二个参数：当前目录下的权重文件

      运行后生成pyramidbox_1000.onnx，该模型只支持bs1。

   3. 使用ATC工具将ONNX模型转OM模型。

       1. 配置环境变量。

          ```shell
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

          ```shell
          atc --framework=5 --model=pyramidbox_1000.onnx --input_format=NCHW --input_shape="image:1,3,1000,1000" --output=pyramidbox_bs1 --log=error --soc_version=Ascend${chip_name} --precision_mode=force_fp32 --fusion_switch_file=fusion_switch.cfg
          ```

          - 参数说明：

            -   --model：为ONNX模型文件。
            -   --framework：5代表ONNX模型。
            -   --output：输出的OM模型。
            -   --input\_format：输入数据的格式。
            
            -   --input\_shape：输入数据的shape。
            -   --log：日志级别。
            -   --soc\_version：处理器型号。
            -   --precision_mode：精度模式
            -   --fusion_switch_file=fusion_switch.cfg：关闭算子融合的配置文件
            

   运行成功后生成pyramidbox_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        ```shell
        mkdir result11 result22
        
        python -m ais_bench --model ${model_path}/pyramidbox_bs1.om --input=${dataset_path}/bs1_data_1/ --outfmt=BIN --output=${output_path} --output_dirname=${output_dir_name}

        python -m ais_bench --model ${model_path}/pyramidbox_bs1.om --input=${dataset_path}/bs1_data_2/ --outfmt=BIN --output=${output_path} --output_dirname=${output_dir_name}
        
        说明：由于预处理后的数据集有两个，所以此脚本需要运行两次
        ```

        -   参数说明：
   
             -   model：om所在的路径。
             -   input：预处理后的所有bin文件。
             -   outfmt：输出格式，此处默认为BIN。
             -   output：输出文件路径
             -   output_dirname: 输出保存文件夹,可以对应取名 result11 和 result22

        推理后的输出默认在当前目录`result11` 和 `result22`下。
   


   3. 处理目录下的bin文件
   
      ```shell
      python3.7 convert.py ./result11 ./result/result11
      python3.7 convert.py ./result22 ./result/result22
      ```
      
      * 参数说明
        * 第一个参数：生成的文件（需要改成对应日期）
        * 第二个参数：生成的二级目录所在文件夹
   
3. 精度验证。

   经过后处理脚本和精度评估文件，此处evaluation文件中目标文件名(../output_1280)需要和后处理输出一致。运行结束后会显示精度结果。

   ```shell
   cd ./pyramidbox
   python3.7 pyramidbox_postprocess.py ./evaluate/ground_truth/wider_face_val.mat ./images ./output_1280 ./result/result11 ./result/result22
   
   cd ./pyramidbox/evaluate
   python3.7 evaluation.py
   ```

   * 后处理参数说明
     * 第一个参数：mat文件目录
     * 第二个参数：图片存储目录
     * 第三个参数：后处理结果保存路径
     * 第四个参数：处理后的bin文件目录1
     * 第五个参数：处理后的bin文件目录2

4. 性能验证。

   可使用ais_bench推理工具的纯推理模式验证om模型的性能，参考命令如下：

   ```shell
   python -m ais_bench --model ${model_path}/pyramidbox_bs1.om --loop 20 --batchsize 1
   ```

   -   参数说明：

        -   model：om所在的路径
        -   loop：循环次数
        -   batchsize：模型batch size

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| 310P3 | 1 | Widerface | Easy   Val AP: 0.9629268827693285<br/>Medium Val AP: 0.9538798956286163<br/>Hard   Val AP: 0.8808383584682273 | 8.9 |