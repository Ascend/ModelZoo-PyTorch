# RCF模型-推理指导


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
RCF（Richer Convolutional Features）通过自动学习将所有卷积层的信息组合起来，从而能够获得不同尺度的更加精细的特征。RCF包含基于VGG 16 Backbone的五个层级的特征提取架构，更加充分利用对象的多尺度和多级信息来全面地执行图像到图像的预测。RCF 不只是使用了每个层级的输出，而是使用了每个层级中所有卷积层的输出进行融合（Conv + sum）后，作为边缘检测的输入。

- 参考实现：

  ```
  url=https://github.com/meteorshowers/RCF-pytorch
  commit_id=6e039117c0b36128febcbe2609b27cc89740a3a8
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | image    | RGB_FP32 | batchsize x 3 x 321 x 481 | NCHW         |
  | image    | RGB_FP32 | batchsize x 3 x 481 x 321 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小               | 数据排布格式 |
  | -------- | -------- | ------------------ | ------------ |
  | output1  | FLOAT32  | batchsize x 3 x 321 x 481 | ND           |
  | output1  | FLOAT32  | batchsize x 3 x 481 x 321 | ND           |
  | output2  | FLOAT32  | batchsize x 3 x 321 x 481 | ND           |
  | output2  | FLOAT32  | batchsize x 3 x 481 x 321 | ND           |
  | output3  | FLOAT32  | batchsize x 3 x 321 x 481 | ND           |
  | output3  | FLOAT32  | batchsize x 3 x 481 x 321 | ND           |
  | output4  | FLOAT32  | batchsize x 3 x 321 x 481 | ND           |
  | output4  | FLOAT32  | batchsize x 3 x 481 x 321 | ND           |
  | output5  | FLOAT32  | batchsize x 3 x 321 x 481 | ND           |
  | output5  | FLOAT32  | batchsize x 3 x 481 x 321 | ND           |
  | output6  | FLOAT32  | batchsize x 3 x 321 x 481 | ND           |
  | output6  | FLOAT32  | batchsize x 3 x 481 x 321 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.7.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。
   ```shell
    git clone https://github.com/meteorshowers/RCF-pytorch
    cd RCF-pytorch
    git reset --hard 6e039117c0b36128febcbe2609b27cc89740a3a8
    cp ../RCF.diff ./
    git apply --check RCF.diff
    git apply RCF.diff
    cd ..  
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```
3. 获取后处理源码
    ```shell
    git clone https://github.com/Walstruzz/edge_eval_python.git
    cd edge_eval_python
    git reset --hard 3e2a532ab939f71794d4cc3eb74cbf0797982b4c
    cd cxx/src
    source build.sh
    mv ../__init__.py impl/
    cd ..
    ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型支持BSDS500数据。官网获取数据集后，解压并上传数据集到主目录下。目录结构如下：

   ```
   RCF
   ├── BSR
        ├── bench
        ├── BSDS500
        └── documentation
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   1. 执行rcf_preprocess.py脚本。

        ```
        python3 rcf_preprocess.py --src_dir BSR/BSDS500/data/images/test --save_path images_npy
        ```
        - 参数说明：
            - src_dir：数据集路径。
            - save_path：预处理的数据储存路径。

    2. 运行gen_dataset_info.py脚本生成图片信息文件。
        ```shell
        python3 gen_dataset_info.py jpg BSR/BSDS500/data/images/test rcf_prep.info
        ```
        - 参数说明
            - 第一个参数为生成的数据集文件格式
            - 第二个参数为原始数据文件相对路径
            - 第三个参数为生成的info文件名


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       [RCF模型权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/RCF/PTH/RCFcheckpoint_epoch12.pth)

   2. 导出onnx文件。

      1. 使用rcf_pth2onnx.py导出onnx文件。该模型只能在GPU上进行转换。

        ```shell
        python3 rcf_pth2onnx.py --pth_path RCFcheckpoint_epoch12.pth --onnx_name rcf.onnx
        ```
        - 参数说明：
            - pth_path：权重文件路径.
            - onnx_name：导出的onnx模型路径。

         获得rcf.onnx文件。

      2. 优化ONNX文件。
           安装auto-optimizer工具。请访问[auto-optimizer改图工具](https://gitee.com/ascend/msadvisor/tree/master/auto-optimizer)代码仓，根据readme文档进行工具安装。

         ```
         python3 -m auto_optimizer optimize rcf.onnx rcf_new.onnx -k 4
         ```

         获得rcf_new.onnx文件。

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

         ```shell
          atc --framework 5 --model rcf_new.onnx --output rcf_bs1 \
              --input_shape "image:1,3,-1,-1" --soc_version Ascend${chip_name} \
              --input_format ND --dynamic_dims "321,481;481,321"
         ```

         - 参数说明：

           - model：为ONNX模型文件。
           - framework：5代表ONNX模型。
           - output：输出的OM模型。h、w取值为321、481和481、321。
           - input\_shape：输入数据的shape。
           - input_format：输入数据的格式。
           - soc\_version：处理器型号。
           - dynamic_dims：设置输入图片的动态分辨率参数。适用于执行推理时，每次处理图片宽和高不固定的场景。

           运行成功后生成`rcf_bs1.om`模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3 -m ais_bench --model rcf_bs1.om --input images_npy --output ./ --output_dirname results_bs1 --auto_set_dymdims_mode 1
        ```

        -   参数说明：

             -   model：om模型路径。
             -   input：输入数据存储路径。
             -   output：输出数据存储路径。
             -   output_dirname：输出数据子目录。
             -   auto_set_dymdims_mode：设置自动匹配动态shape

        推理后的输出默认在当前目录results_bs1下。


   3. 精度验证。

      1. 运行rcf_postprocess.py脚本，将bin文件转存为mat文件。

            ```
            python3 rcf_postprocess.py --imgs_dir BSR/BSDS500/data/images/test --bin_dir results_bs1 --om_output om_out
            ```

            - 参数说明：
                - imgs_dir：数据集图片路径。
                - bin_dir：om模型输出的bin文件路径。
                - om_output：mat文件存储路径。

        2. 运行main.py脚本，计算精度。
            ```shell
            cd edge_eval_python
            python3 main.py --alg "RCF" --model_name_list "rcf" --result_dir ../om_out \
                --save_dir ../eval_result --gt_dir ../BSR/BSDS500/data/groundTruth/test \
                --key om_result --file_format .mat --workers -1
            ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ------------ | ---------- | --------------- |
|   310P3   |        1         |   BSDS500    |   ODS：0.798，OIS：0.817   |      93.76      |