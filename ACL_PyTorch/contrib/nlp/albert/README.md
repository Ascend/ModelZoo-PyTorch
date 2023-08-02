# Albert模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

ALBERT是BERT 的“改进版”，主要通过通过Factorized embedding parameterization和Cross-layer parameter sharing两大机制减少参数量，得到一个占用较小的模型，对实际落地有较大的意义，不过由于其主要还是减少参数，不影响推理速度。

  ```
  url=https://github.com/lonePatient/albert_pytorch
  branch=master
  commit_id=46de9ec
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据       | 数据类型 | 大小                      | 数据排布格式 |
  | --------       | -------- | ------------------------- | ------------ |
  | input_ids      | INT64    | batchsize x seq_len       | ND           |
  | attention_mask | INT64    | batchsize x seq_len       | ND           |
  | token_type_ids | INT64    | batchsize x seq_len       | ND           |

  说明：该模型默认的seq_len为128

- 输出数据

  | 输出数据 | 大小               | 数据类型 | 数据排布格式 |
  | -------- | --------           | -------- | ------------ |
  | output   | batch_size x class | FLOAT32  | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                            | 版本    | 环境准备指导                                                                                          |
| ------------------------------------------------------------    | ------- | ------------------------------------------------------------                                          |
| 固件与驱动                                                      | 23.0.RC2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                            | 6.3.RC2 | -                                                                                                     |
| Python                                                          | 3.7.5   | -                                                                                                     |
| PyTorch                                                         | 1.5.0+ | -                                                                                                     |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master         # 切换到对应分支
   cd ACL_PyTorch/contrib/nlp/albert              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   git clone https://gitee.com/ascend/msadvisor && cd msadvisor && git checkout master
   cd auto-optimizer && python3 -m pip install .
   cd ../..
   ```

2. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/lonePatient/albert_pytorch.git
   cd albert_pytorch
   git checkout 46de9ec
   patch -p1 < ../albert.patch
   cd ../
   ```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。

   本模型采用[SST-2数据集](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip)，解压到 `albert_pytorch/dataset/SST-2`文件夹下

   数据目录结构请参考：
   ```
   ├──SST-2
    ├──original/
    ├──dev.tsv
    ├──train.tsv
    ├──test.tsv
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。该模型数据预处理需要加载模型，所以需要先下载权重文件：

   获取[预训练权重文件](https://drive.google.com/open?id=1byZQmWDgyhrLpj8oXtxBG6AA52c8IHE-)，并解压到albert_pytorch/prev_trained_model/albert_base_v2。

   下载[训练好的模型](https://pan.baidu.com/s/1G5QSVnr2c1eZkDBo1W-uRA )（提取码：mehp ）并解压到albert_pytorch/outputs/SST-2。

   执行“Albert_preprocess.py”脚本，完成预处理。

   ```
   python3 Albert_preprocess.py --pth_dir=./albert_pytorch/outputs/SST-2/ --data_dir=./albert_pytorch/dataset/SST-2/ --save_dir ./preprocessed_data_seq128 --max_seq_length 128
   ```
   - 参数说明：

     --pth_dir: 模型权重所在路径

     --data_dir：原始数据集所在路径

     --save_dir: 预处理数据保存路径, 其中gt_label保存在 `${save_dir}/label.npy`
     
     --max_seq_length: 对应的seq长度，默认为128，支持：16/32/64/128


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
   
      数据预处理阶段已经完成模型权重下载。

   2. 导出onnx文件。

      1. 使用脚本导出onnx文件。

         运行“Albert_pth2onnx.py”脚本。

         ```
         # pth转换为ONNX，此处以seq128/bs32为例
         python3 ./Albert_pth2onnx.py --batch_size=32 --pth_dir=./albert_pytorch/outputs/SST-2/ --data_dir=./albert_pytorch/datasets/SST-2/ --onnx_dir=./outputs/ --max_seq_length=128
         ```

         - 参数说明：

           --batch_size: 导出模型batchsize。

           --pth_dir：权重所在路径。
           
           --data_dir: 数据集所在路径。

           --onnx_dir: 输出onnx文件所在目录。
           
           --max_seq_length: 模型对应seq，默认为128，支持：16/32/64/128。

         获得outputs/albert_seq128_bs32.onnx文件。

      2. 优化ONNX文件。

         静态ONNX模型:

         ```
         # 以seq128/bs32为例
         python3 -m onnxsim ./outputs/albert_seq128_bs32.onnx ./outputs/albert_seq128_bs32_sim.onnx
         python3 opt_onnx.py --input_file ./outputs/albert_seq128_bs32_sim.onnx --output_file ./outputs/albert_seq128_bs32_opt.onnx
         ```
         
         动态ONNX模型:
         
         ```
         # bs: [4, 8, 16, 32]
         python3 fix_onnx2unpad.py --input_file ./outputs/albert_seq128_bs${bs}.onnx --output_file ./outputs/albert_seq128_bs${bs}_unpad.onnx
         ```

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
          # 使能transformer加速库：动态Unpad方案必需
          source ${ASCENDIE_HOME}/set_env.sh
         ```

         > **说明：**
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3 （自行替换）
         回显如下：
         +-------------------|-----------------|------------------------------------------------------+
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
      
         静态模型转化:
     
         ```
         # 以seq128/bs32为例
         atc --input_format=ND --framework=5 --model=./outputs/albert_seq128_bs32_opt.onnx --output=./outputs/albert_seq128_bs32 --log=error --soc_version=${chip_name} --input_shape="input_ids:32,128;attention_mask:32,128;token_type_ids:32,128" --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --optypelist_for_implmode：需要指定精度模式的算子。
           -   --op_select_implmode：特定算子需要采取的精度模式。

           运行成功后生成albert_seq128_b32.om模型文件。

         对于`seq16`对应的模型，ATC命令有所区别，如下：
        
         ```
         # 以seq16/bs64为例
         atc --input_format=ND --framework=5 --model=./outputs/albert_seq16_bs64_opt.onnx --output=./outputs/albert_seq16_bs64 --log=error --soc_version=${chip_name} --input_shape="input_ids:64,16;attention_mask:64,16;token_type_ids:64,16" --op_precision_mode=precision.ini
         ```

         - 额外参数说明：

           -   --op_precision_mode：算子精度模式配置输入。
           
         动态模型转化:
     
         ```
         atc --input_format=ND --framework=5 --model=./outputs/albert_seq128_bs${bs}_unpad.onnx --output=./outputs/albert_seq128_bs${bs}_unpad --log=error --soc_version=${chip_name} --input_shape="input_ids:-1,128;attention_mask:-1,128;token_type_ids:-1,128" --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --optypelist_for_implmode：需要指定精度模式的算子。
           -   --op_select_implmode：特定算子需要采取的精度模式。

           运行成功后生成`albert_seq128_unpad_${os}_${arch}.om`模型文件。


2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。
   
        静态模型推理:

        ```
        # 以bs32为例
        python3 -m ais_bench --model outputs/albert_seq128_bs32.om --input ./preprocessed_data_seq128/input_ids,./preprocessed_data_seq128/attention_mask,./preprocessed_data_seq128/token_type_ids --output results --output_dirname seq128_bs32 --outfmt NPY --batchsize 32
        ```
        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --output_dirname：输出文件名。
             -   --device：NPU设备编号。
             -   --outfmt: 输出数据格式。
             -   --batchsize：推理模型对应的batchsize。


        推理后的输出默认在当前目录results/seq128_bs32下。
        
        动态模型推理:
        
        ```
        # 以bs32为例
        python3 -m ais_bench --model outputs/albert_seq128_bs32_unpad_${os}_${arch}.om --input ./preprocessed_data_seq128/input_ids,./preprocessed_data_seq128/attention_mask,./preprocessed_data_seq128/token_type_ids --output results_dynamic --output_dirname bs32 --outfmt NPY --dymShape "input_ids:32,128;attention_mask:32,128;token_type_ids:32,128" --outputSize 1000000
        ```
        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --output_dirname：输出文件名。
             -   --device：NPU设备编号。
             -   --outfmt: 输出数据格式。
             -   --dymShape：动态模型推理输入shape。
             -   --outputSize: 动态模型推理输出buffer大小。

        推理后的输出默认在当前目录results_dynamic/bs32下。

   3.  精度验证。

      调用Albert_postprocess.py脚本与数据集标签比对，获得Accuracy数据。

      ```
      python3 Albert_postprocess.py --result_dir results/seq128_bs32 --label_path preprocessed_data_seq128/label.npy
      ```

      -   参数说明：

        --result_dir：生成推理结果所在路径。

        --label_path：GT label文件所在路径。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

seq128对应的精度性能如下：

精度:

| 模型方案  | device   | ACC(seq128)   |
| --------- | -------- | ------------- |
| 静态      | 基准     | 92.8%         |
| 静态      | 310      | 92.7%         |
| 静态      | 310P     | 92.8%         |
| 动态      | 310P     | 92.8%         |

静态模型性能：

| 模型        | 310性能   | 310P3性能 |
| :------:    | :------:  | :------:  |
| Albert bs1  | 231.39fps | 763fps    |
| Albert bs4  |           | 1148fps   |
| Albert bs8  |           | 1321fps   |
| Albert bs16 | 300.83fps | 1350fps   |
| Albert bs32 |           | 1320fps   |
| Albert bs64 |           | 1330fps   |

动态模型性能（数据集推理）：

| 模型        | 310P3性能 |
| :------:    | :------:  |
| Albert bs4  | 535fps    |
| Albert bs8  | 953fps    |
| Albert bs16 | 1518fps   |
| Albert bs32 | 2195fps   |
| Albert bs64 | 2346fps   |


其他seq精度性能结果如下(不同seq模型：展示bs1和最优bs精度/性能)：

| seq | batch size | pth精度 | 310P精度 | 310P性能 |
|-----|------------|---------|----------|----------|
|  16 |          1 | 58.5%   | 58.6%    | 1180fps  |
|  16 |         64 | -       | -        | 9775fps  |
|  32 |          1 | 79.8%   | 80.4%    | 926fps   |
|  32 |         64 | -       | -        | 5843fps  |
|  64 |          1 | 92.7%   | 92.8%    | 582fps   |
|  64 |         32 | -       | -        | 2937fps  |
