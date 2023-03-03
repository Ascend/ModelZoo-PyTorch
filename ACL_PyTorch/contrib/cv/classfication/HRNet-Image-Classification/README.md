# HRNet模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

HRNet（High-Resolution Net）是针对2D人体姿态估计（Human Pose Estimation或Keypoint Detection）任务提出的，并且该网络主要是针对单一个体的姿态评估（即输入网络的图像中应该只有一个人体目标）。
- 参考实现：

  ```
  url=https://github.com/HRNet/HRNet-Image-Classification
  commit_id=f130a24bf09b7f23ebd0075271f76c4a188093b2
  code_path=https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/classfication/HRNet-Image-Classification
  model_name=HRNet
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小               | 数据排布格式 |
  | -------- |------------------| -------- | ------------ |
  | output1  | FLOAT32  | batchsize x 1000 | ND           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本      | 环境准备指导                                                 |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/HRNet/HRNet-Image-Classification.git
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）


   该模型使用[ImageNet官网](http://www.image-net.org/)的5万张验证集进行测试，图片与标签分别存放在/local/HRNet/imagenet/val与/local/HRNet/imagenet/val_label.txt。
   ```
   imagenet
   ├── val_label.txt    //验证集标注信息       
   └── val             // 验证集文件夹
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   执行imagenet_torch_preprocess.py脚本，完成预处理。

   ```
   python3.7 imagenet_torch_preprocess.py hrnet  /local/HRNet/imagenet/val ./prep_dataset

   ```
   
   - 参数说明：
   
     /local/HRNet/imagenet/val，原始数据验证集（.jpeg）所在路径。
         
     ./prep_dataset，输出的二进制文件（.bin）所在路径。



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      [HRNet预训练pth权重文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/HrNet/PTH/model_best.pth.tar)

      ```
      进入网页点击下载模型，压缩包中有model_best.pth.tar的权重文件
      ```
      
   2. 导出onnx文件。

      1. 使用hrnet_pth2onnx.py脚本。

         运行hrnet_pth2onnx.py脚本。

         ```
         python3.7 hrnet_pth2onnx.py --cfg ./HRNet-Image-Classification/experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --input model_best.pth.tar --output hrnet_w18.onnx
         ```

         获得hrnet_w18.onnx文件。

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
         atc --framework=5 --model=./hrnet_w18.onnx --input_format=NCHW --input_shape="image:{batch size},3,224,224" --output=hrnet_bs{batch size} --log=debug --soc_version=Ascend310P3
         示例
         atc --framework=5 --model=./hrnet_w18.onnx --input_format=NCHW --input_shape="image:16,3,224,224" --output=hrnet_bs16 --log=debug --soc_version=Ascend310P3
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成hrnet_bs16.om模型文件，batch size为1、4、8、32、64的修改对应的batch size的位置即可。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        python3 -m ais_bench --model ./hrnet_bs{batch size}.om --input ./prep_dataset/ --output ./output --output_dirname subdir --outfmt 'TXT' --batchsize {batch size}
        示例
        python3 -m ais_bench --model ./hrnet_bs16.om --input ./prep_dataset/ --output ./output --output_dirname subdir --outfmt 'TXT' --batchsize 16
        ```

        -   参数说明：

             -   model：需要推理om模型的路径。
             -   input：模型需要的输入bin文件夹路径。
             -   output：推理结果输出路径。
             -   outfmt：输出数据的格式。
             -   output_dirname:推理结果输出子文件夹。

        推理后的输出默认在当前目录output的subdir下。

   3. 精度验证。

      调用imagenet_acc_eval.py脚本推理结果与label比对，可以获得Accuracy Top5数据，结果保存在result.json中。

      ```
      python3.7 imagenet_acc_eval.py ./output/subdir /local/HRNet/imagenet/val_label.txt ./ result.json
      ```

      - 参数说明：

        - ./output/subdir/：为生成推理结果所在路径  

        - /local/HRNet/imagenet/val_label.txt：为标签数据所在路径

        - ./ result.json：结果保存路径

   4. 性能验证。

     可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

       ```
       python3.7 -m ais_bench --model=./hrnet_bs{batch size}.om --loop=1000 --batchsize={batch size}
       示例
       python3.7 -m ais_bench --model=./hrnet_bs16.om --loop=1000 --batchsize=16
       ```

     - 参数说明：
       - --model：需要验证om模型所在路径
       - --batchsize：验证模型的batch size，按实际进行修改



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集 | 精度                    | 性能   |
| --------- |------------| ---------- |-----------------------|------|
|   310P3        | 1          |  ImageNet          | 76.51/Top1 93.22/Top5 | 646  |
|   310P3        | 4          |  ImageNet          | 76.51/Top1 93.22/Top5 | 1510 |
|   310P3        | 8          |  ImageNet          | 76.51/Top1 93.22/Top5 | 1778 |
|   310P3        | 16         |  ImageNet          | 76.51/Top1 93.22/Top5 | 2250 |
|   310P3        | 32         |  ImageNet          | 76.51/Top1 93.22/Top5 | 1539 |
|   310P3        | 64         |  ImageNet          | 76.51/Top1 93.22/Top5 | 1320 |