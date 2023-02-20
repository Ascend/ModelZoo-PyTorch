# ViT_base模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

`Transformer` 架构已广泛应用于自然语言处理领域。本模型的作者发现，Vision Transformer（ViT）模型在计算机视觉领域中对CNN的依赖不是必需的，直接将其应用于图像块序列来进行图像分类时，也能得到和目前卷积网络相媲美的准确率。

- 参考实现：

  ```
  url=https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
  mode_name=vit_base_patch32_224
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32  | batchsize x 3 x 224 x 224 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小                  | 数据排布格式 |
  | -------- | -------- | --------              | ------------ |
  | output   | FLOAT32  | batchsize x num_class | ND           |

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
   cd ACL_PyTorch/contrib/cv/ViT_base              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

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
   mkdir -p prep_dataset
   python3 Vit_base_preprocess.py --data-path ImageNet/val/ --store-path ./prep_dataset
   ```

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   获取 B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz权重。具体下载链接可参考：https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py

   1. 导出onnx文件。

      1. 使用以下脚本导出onnx文件:

         ```
         # 以bs1为例
         mkdir -p models/onnx
         python3 Vit_base_pth2onnx.py --batch_size 1 --model_path B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz --save_dir models/onnx --model_name vit_base_patch32_224
         ```

         获得vit_base_bs1.onnx文件。

      2. 优化ONNX文件。

         ```
         # bs1为例
         python3 -m onnxsim models/onnx/vit_base_bs1.onnx models/onnx/vit_base_bs1_sim.onnx
         # 输入参数: 1.原始模型文件路径 2.优化模型文件路径 3.batchsize
         python3 opt_vit.py models/onnx/vit_base_bs1_sim.onnx models/onnx/vit_base_bs1_opt.onnx 1
         ```

         获得vit_base_bs1_opt.onnx文件。

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
         # bs1为例
         mkdir -p models/om
         atc --framework=5 --model=models/onnx/vit_base_bs1_opt.onnx --output=models/om/vit_base_bs1 --input_format=NCHW --input_shape="input:1,3,224,224" --log=debug --soc_version=${chip_name} --enable_small_channel=1 --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --precision_model: 精度模式。
           -   --modify_mixlist: 算子精度配置文件。

           运行成功后生成vit_base_bs1.om模型文件。

2. 开始推理验证。<u>***根据实际推理工具编写***</u>

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```
        # 以bs1为例
        mkdir -p outputs/bs1
        python3 -m ais_bench --model models/om/vit_base_bs1.om --input prep_dataset/ --output outputs/bs1 --device 1 --batchsize 1
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --device：NPU设备编号。
             -   --batchsize: 模型对应batchsize。


        推理后的输出默认在当前目录outputs/bs1下。


   3. 精度验证。

      调用脚本与GT label，可以获得精度数据:

      ```
      # 以bs1为例
      python3 Vit_base_postprocess.py --save_path result_bs1.json --input_dir ./outputs/bs1/${timestamp} --label_path ImageNet/val_label.txt
      ```

      - 参数说明：

        - --input_dir：为生成推理结果所在路径

        - --label_path：为标签数据路径

        - --save_path: 结果保存路径


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

   调用ACL接口推理计算，性能参考下列数据。

   1.精度对比

| 模型                     | 仓库pth精度 | 310离线推理精度 | 310P离线推理精度 |
|--------------------------|-------------|-----------------|------------------|
| ViT_base_patch32_224 bs1 | top1:80.724 | top1:80.714     | top1:80.65       |
| ViT_base_patch32_224 bs8 | top1:80.724 | top1:80.714     | top1:80.65       |


   2.性能对比

| Throughput |      310 |      310P | 310P/310 |
|------------|----------|-----------|----------|
| bs1        | 321.2008 |  412.4932 |     1.28 |
| bs4        | 444.8147 | 1074.8210 |     2.42 |
| bs8        | 433.9089 | 1319.7835 |     3.04 |
| bs16       | 424.4384 | 1511.8729 |     3.56 |
| bs32       | 413.3992 | 1489.4850 |     3.60 |
| bs64       | 400.4219 | 1301.1106 |     3.25 |
| 最优batch  | 444.8147 | 1511.8729 |     3.40 |

