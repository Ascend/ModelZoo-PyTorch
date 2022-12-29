# Nasnetlarge模型-推理指导


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

SPNASNet100是通过网络搜索技术得到的精度与效率权衡的卷积神经网络，用于图像分类任务。


- 参考实现：

  ```
  url=https://github.com/Cadene/pretrained-models.pytorch.git
  commit_id=b8134c79b34d8baf88fe0815ce6776f28f54dbfe
  code_path=ACL_PyTorch/contrib/cv/detection/Nasnetlarge
  model_name=Nasnetlarge
  ```
  



## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                     | 数据排布格式 |
  | -------- | -------- | ------------------------ | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 331x 331 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | output1  | FP32     | Batchsize x 1000 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/Cadene/pretrained-models.pytorch 
   cd pretrained-models.pytorch
   python3 setup.py install
   ```
   
2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```
   
   >**注**：在获取源码前建议先执行2.安装依赖，获取源码中python3 setup.py install 需要用到一些依赖文件。已在requirements.txt中列出。

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持[ImageNet 50000](https://gitee.com/link?target=http%3A%2F%2Fwww.image-net.org)张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（如：/opt/npu/）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。目录结构如下：

   ```
   ├── ImageNet
   ├── val
   ├── val_label.txt  
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   将原始数据（.jpeg）转化为二进制文件（.bin）。

   执行preprocess_img.py脚本，完成预处理。

   ```
   python3 preprocess_img.py ${datasets} ${prep_dataset}
   ```

   + 参数说明：
     + ${datasets}：原始数据验证集（.jpeg）所在路径,例如：/opt/npu/ImageNet/val。
     + ${prep_dataset}：输出的二进制文件（.bin）所在路径。

   每个图像对应生成一个二进制文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从源码包中获取权重文件[nasnetalarge-a1897284.pth](http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth)。

       ```
       wget http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth
       ```

   2. 导出onnx文件。

      1. 使用nasnetlarge_pth2onnx.py导出onnx文件。

         运行nasnetlarge_pth2onnx.py脚本。

         ```
         python3 nasnetlarge_pth2onnx.py nasnetalarge-a1897284.pth nasnetlarge.onnx
         ```

         获得nasnetlarge.onnx文件。

      2. 优化ONNX文件。

         使用onnxsim，生成不同batch size的onnx_sim模型文件

         ```
         python3 -m onnxsim --input-shape="1,3,331,331" nasnetlarge.onnx nasnetlarge_sim1.onnx
         ```
         
         获得nasnetlarge_sim1.onnx文件。
         
      3. 不同bs算子融合优化。

         ```
         git clone https://gitee.com/ascend/msadvisor.git
         cd /msadvisor/auto-optimizer
         pip3 install -r requirements.txt
         python3 setup.py install 
         cd ../..
         python3 -m auto_optimizer opt nasnetlarge_sim1.onnx nasnetlarge_sim1_merge.onnx -k KnowledgeMergeConsecutiveSlice
         ```
      
          获得nasnetlarge_sim1_merge.onnx文件。

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
         atc --framework=5 --model=nasnetlarge_sim1_merge.onnx --input_format=NCHW --input_shape="image:1,3,331,331" --output=nasnetlarge_sim1_merge.onnx --log=debug --soc_version=Ascend${chip_name}
         ```
         
         - 参数说明：
         
           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           
         
         运行成功后生成nasnetlarge_sim1_merge.onnx模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3 -m ais_bench --model nasnetlarge_sim1_merge.onnx --input ./prep_dataset --output ./result/ --output_dirname bs1 --outfmt TXT --batchsize 1
        ```

        -   参数说明：

             -   --model：om模型。
             -   --input：预处理数据集路径。
             -   --output：推理结果所在路径。
             -   ----output_dirname：推理结果输出子文件夹。可选参数。与参数output搭配使用，单独使用无效。设置该值时输出结果将保存到 output/output_dirname文件夹中 。
             -   --outfmt：推理结果文件格式。
             -   --batchsize：不同的batchsize。
   
        推理后的输出默认在当前目录result下。
   
   
   3. 精度验证。
   
      调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据。，结果保存在result.json中。
   
      ```
       python3 imagenet_acc_eval.py result/bs1/ /opt/npu/imagenet/val_label.txt ./ result.json
      ```
   
      - 参数说明：
   
        - result/bs1/：为生成推理结果所在路径  
        - /opt/npu/imagenet/val_label.txt：标签数据。
        - ./ ： 生成结果文件路径
        - result.json：生成结果文件名称。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

精度对比：

| Model     | Nasnetlarge                |
| --------- | -------------------------- |
| 标杆精度  | top1：82.56%  top5：96.08% |
| 310P3精度 | top1：82.5%   top5：96.02% |

性能对比：

| 芯片型号 | Batch Size   | 数据集 | 性能 |
| --------- | ---------------- | ---------- | --------------- |
| 310P3 | 1 | ILSVRC2012 | 153.316 |
| 310P3 | 4 | ILSVRC2012 | 175.744 |
| 310P3 | 8 | ILSVRC2012 | 162.623 |
| 310P3 | 16 | ILSVRC2012 | 146.502 |
| 310P3 | 32 | ILSVRC2012 | 135.831 |
| 310P3 | 64 | ILSVRC2012 | 113.459 |