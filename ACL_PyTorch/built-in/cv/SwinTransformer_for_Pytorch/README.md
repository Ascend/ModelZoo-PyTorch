# SwinTransformer模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

`Swin Transformer`是在`Vision Transformer`的基础上使用滑动窗口（shifted windows, SW）进行改造而来。它将`Vision Transformer`中固定大小的采样快按照层次分成不同大小的块（Windows），每一个块之间的信息并不共通、独立运算从而大大提高了计算效率。

- 参考实现：

  ```
  url=https://github.com/rwightman/pytorch-image-models
  mode_name=SwinTransformer
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  说明：官方SwinTransformer仓的输入图片宽高相等，具体尺寸可参考配置：如swin_base_patch4_window12_384配置对应尺寸为384。

  | 输入数据 | 数据类型 | 大小                           | 数据排布格式 |
  | -------- | -------- | -------------------------      | ------------ |
  | image    | FLOAT16 | batchsize x 3 x height x width | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小                  | 数据排布格式 |
  | -------- | -------- | --------              | ------------ |
  | logits   | FLOAT16  | batchsize x num_class | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动:

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0+ | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master         # 切换到对应分支
   cd ACL_PyTorch/built-in/cv/SwinTransformer_for_Pytorch              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

   安装 [ait](https://gitee.com/ascend/ait/tree/master/ait)。

## 准备数据集<a name="section183221994411"></a>


   本模型使用ImageNet 50000张图片的验证集，请前往[ImageNet官网](https://image-net.org/download.php)下载数据集:

    ```
    ├── ImageNet
    |   ├── val
    |   |    ├── ILSVRC2012_val_00000001.JPEG
    │   |    ├── ILSVRC2012_val_00000002.JPEG
    │   |    ├── ......
    |   ├── val_label.txt
    ```

   执行预处理脚本:

   ```
   python3 preprocess.py --img_size 384 --input_dir /opt/npu/imagenet/val --out_dir ./preprocessed_data
   ```

   - 参数说明：

     -   --img_size：模型输入尺寸：384/224。
     -   --input_dir：原始数据文件夹地址。
     -   --out_dir：预处理数据输出路径。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   获取权重文件：[swin_base_patch4_window12_384_22kto1k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth)，其他模型权重下载地址可参考：[Swin-Transformer(microsoft)](https://github.com/microsoft/Swin-Transformer)

   1. 导出onnx文件。

      1. 使用以下脚本导出onnx文件:

         ```
         python3 pth2onnx.py --input_path swin_base_patch4_window12_384_22kto1k.pth --out_path models/onnx/swin_base_patch4_window12_384_bs${bs}.onnx --model_name swin_base_patch4_window12_384 --batch_size ${bs}
         ```
         
         - 参数说明：

           -   --input_path：模型权重文件所在路径。
           -   --out_path：输出onnx文件所在路径。
           -   --model_name：模型名。
           -   --batch_size：模型对应batch_size。

         获得swin_base_patch4_window12_384_bs${bs}.onnx文件。

      2. 优化ONNX文件。

         ```
         # 以bs8为例
         python3 -m onnxsim models/onnx/swin_base_patch4_window12_384_bs8.onnx models/onnx/swin_base_patch4_window12_384_bs8.onnx
         python3 opt_onnx.py -i models/onnx/swin_base_patch4_window12_384_bs8.onnx -o models/onnx/swin_base_patch4_window12_384_bs8_opt.onnx
         ```

         获得swin_base_patch4_window12_384_bs8_opt.onnx文件。

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

         ```
         # bs8为例
         atc --model=models/onnx/swin_base_patch4_window12_384_bs8_opt.onnx --framework=5 --output=models/om/swin_base_patch4_window12_384_bs8 --input_format=NCHW --log=debug --soc_version=${chip_name} --output_type=FP16 --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance --insert_op_conf aipp.config --enable_small_channel 1
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成swin_base_patch4_window12_384_bs8.om模型文件。

2. 开始推理验证。<u>***根据实际推理工具编写***</u>

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        # 以bs8为例
        mkdir -p outputs/bs8
        python3 -m ais_bench --model models/om/swin_base_patch4_window12_384_bs8.om --input preprocessed_data/ --output outputs/bs8 --device 1 --outfmt NPY
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --device：NPU设备编号。


        推理后的输出默认在当前目录outputs下。


   3. 精度验证。

      调用脚本与GT label，可以获得精度数据:

      ```
      # 以bs8为例
      python3 postprocess.py --input_dir ./outputs/bs8/${timestamp} --label_path ./val_label.txt --save_path ./result_bs8.json
      ```

      - 参数说明：


        - --input_dir：为生成推理结果所在路径
    
        - --label_path：为标签数据路径
    
        - --save_path: 结果保存路径


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

   调用ACL接口推理计算，性能参考下列数据（具体配置为：swin_base_patch4_window12_384）。

   | 芯片型号 | Batch Size | 数据集   | 精度                            | 性能    |
   |----------|------------|----------|---------------------------------|---------|
   | 310P3    | 1          | ImageNet | Top1 Acc: 86.4%;Top5 Acc: 98.0% | 104 fps |
   | 310P3    | 4          | ImageNet | -                               | 121 fps |
   | 310P3    | 8          | ImageNet | -                               | 132 fps |
   | 310P3    | 16         | ImageNet | -                               | 129 fps |
   | 310P3    | 32         | ImageNet | -                               | 129 fps |
   | 310P3    | 64         | ImageNet | -                               | 131 fps |
   | 基准性能 | 1          | ImageNet | Top1 Acc: 86.4%;Top5 Acc: 98.0% | 217 fps |
   | 基准性能 | 4          | ImageNet | -                               | 332 fps |
   | 基准性能 | 8          | ImageNet | -                               | 339 fps |
   | 基准性能 | 16         | ImageNet | -                               | 351 fps |
   | 基准性能 | 32         | ImageNet | -                               | 357 fps |
   | 基准性能 | 64         | ImageNet | -                               | 363 fps |

  其他配置参考性能精度如下(更新bs1/最优bs)：

   | 芯片型号 | config                         | Batch Size | 数据集   | 参考精度                        | NPU精度                           | 性能    |
   |----------|--------------------------------|------------|----------|---------------------------------|-----------------------------------|---------|
   | 310P3    | swin_large_patch4_window12_384 | 1          | ImageNet | Top1 Acc: 87.3%;Top5 Acc: 98.2% | Top1 Acc: 87.27%;Top5 Acc: 98.23% | 51 fps  |
   | 310P3    | -                              | 8          | ImageNet | -                               | -                                 | 76 fps  |
   | 310P3    | swin_large_patch4_window7_224  | 1          | ImageNet | Top1 Acc: 86.3%;Top5 Acc: 97.9% | Top1 Acc: 86.19%;Top5 Acc: 97.83% | 112 fps |
   | 310P3    | -                              | 8          | ImageNet | -                               | -                                 | 204 fps |
