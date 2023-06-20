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
mode_name = [
    vit_base_patch8_224, 
   vit_base_patch16_224, 
   vit_base_patch16_384, 
   vit_base_patch32_224,
   vit_base_patch32_384,
]
```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

   1. 对于 vit_base_patch8_224、vit_base_patch16_224 和 vit_base_patch32_224

      | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
      | -------- | -------- | ------------------------- | ------------ |
      | input    | FLOAT32  | batchsize x 3 x 224 x 224 | NCHW         |

   2. 对于 vit_base_patch16_384 和 vit_base_patch32_384

      | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
      | -------- | -------- | ------------------------- | ------------ |
      | input    | FLOAT32  | batchsize x 3 x 384 x 384 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小                  | 数据排布格式 |
  | -------- | -------- | --------              | ------------ |
  | output   | FLOAT32  | batchsize x num_class | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动:

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.4  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.8.0+  | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```bash
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master                                            # 切换到对应分支
   cd ACL_PyTorch/contrib/cv/classfication/ViT_base     # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```bash
   pip3 install -r requirements.txt
   ```

3. 安装改图工具 auto-optimizer
   ```bash
   git clone https://gitee.com/ascend/msadvisor.git
   cd msadvisor/auto-optimizer
   pip3 install -r requirements.txt
   python3 setup.py install
   cd ../..
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

   ```bash
   # img_size为预处理输出的图像尺寸，可选224或384，需要和模型相对应。
   img_size=224

   python3 Vit_base_preprocess.py --data_path ImageNet/val/ --store_path ./prep_dataset/${img_size} --image_size ${img_size}
   ```
   - 参数说明：
      - --data_path: 数据集路径
      - --store_path: 预处理结果保存路径
      - --image_size: 图像尺寸

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   获取模型权重。
   下载链接可参考：https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py

   模型变体较多，可按需下载。根据下表通过搜索文件名找到对应的权重文件下载地址，下载到当前目录下。

   |            模型变体|                                                                                                文件名|
   |--------------------|------------------------------------------------------------------------------------------------------|
   | vit_base_patch8_224|  B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz|
   |vit_base_patch16_224| B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz|
   |vit_base_patch16_384| B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz|
   |vit_base_patch32_224|B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz|
   |vit_base_patch32_384|  B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz|

   然后将权重文件重命名为```模型变体名称.npz```
   ```bash
   # 以 vit_base_patch8_224 为例
   mv B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz vit_base_patch8_224.npz
   ```

   1. 导出onnx文件。

      1. 使用以下脚本导出onnx文件:

         ```bash
         # bs为Batch Size，可根据需要设置，此处以1为例
         bs=1

         # model_name为模型变体名称，可根据需要设置，此处以 vit_base_patch8_224 为例
         model_name=vit_base_patch8_224

         python3 Vit_base_pth2onnx.py --batch_size ${bs} --model_path ${model_name}.npz --save_dir models/onnx --model_name ${model_name}
         ```
         - 参数说明：
            - --batch_size: 批次大小
            - --model_path: 模型权重npz文件路径
            - --save_dir: 保存onnx文件的目录
            - --model_name: 模型变体名称
         ---
         获得```vit_base_patch8_224_bs1.onnx```文件。

      2. 优化ONNX文件。

         ```bash
         python3 -m onnxsim models/onnx/${model_name}_bs${bs}.onnx models/onnx/${model_name}_bs${bs}_sim.onnx
         
         # 输入参数: 1.原始模型文件路径 2.优化模型文件路径 3.模型变体名称
         python3 opt_vit.py models/onnx/${model_name}_bs${bs}_sim.onnx models/onnx/${model_name}_bs${bs}_opt.onnx ${model_name}
         ```

         获得```vit_base_patch8_224_bs1_opt.onnx```文件。

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

         ```bash
         mkdir -p models/om
         atc --framework=5 --model=models/onnx/${model_name}_bs${bs}_opt.onnx --output=models/om/${model_name}_bs${bs} --input_format=NCHW --input_shape="input:${bs},3,${img_size},${img_size}" --log=debug --soc_version=${chip_name} --enable_small_channel=1 --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
         ```

         - 参数说明：

           - --model：为ONNX模型文件。
           - --framework：5代表ONNX模型。
           - --output：输出的OM模型。
           - --input_format：输入数据的格式。
           - --log：日志级别。
           - --soc_version：处理器型号。
         ---
         运行成功后生成```vit_base_patch8_224_bs1.om```模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

        ```bash
        python3 -m ais_bench --model models/om/${model_name}_bs${bs}.om --input prep_dataset/${img_size} --output outputs/ --output_dir ${model_name}_bs${bs} --device 1
        ```

        -   参数说明：
            - --model：om文件路径
            - --input：输入文件
            - --output：输出目录
            - --device：NPU设备编号

         ---
        推理后的输出默认在当前目录```outputs/${model_name}_bs${bs}```下。


   3. 精度验证。

      调用脚本与GT label，可以获得精度数据:

      ```bash
      python3 Vit_base_postprocess.py --save_path result_${model_name}_bs${bs}.json --input_dir ./outputs/${model_name}_bs${bs} --label_path ImageNet/val_label.txt
      ```

      - 参数说明：
         - --input_dir：为生成推理结果所在路径
         - --label_path：为标签数据路径
         - --save_path: 结果保存路径

   4. 执行纯推理验证性能。
      ```bash
      python3 -m ais_bench --model models/om/${model_name}_bs${bs}.om --device 1 --loop 100
      ```
        -   参数说明：
            - --model：om文件路径
            - --device：NPU设备编号
            - --loop: 纯推理次数
      


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>


|芯片型号|            模型变体|   Batch Size|  数据集|    参考精度|    NPU精度|性能(fps)|
|:------:|:------------------:|:-----------:|:------:|:----------:|:---------:|:-------:|
|   310P3| ViT_base_patch8_224|            1|ImageNet| top1: 85.80|top1: 85.58|    76.42|
|   310P3| ViT_base_patch8_224| 8 (最优性能)|ImageNet| top1: 85.80|top1: 85.58|    98.48|
|   310P3|ViT_base_patch16_224|            1|ImageNet| top1: 84.53|top1: 84.16|   342.34|
|   310P3|ViT_base_patch16_224|16 (最优性能)|ImageNet| top1: 84.53|top1: 84.16|   660.64|
|   310P3|ViT_base_patch16_384|            1|ImageNet| top1: 86.01|top1: 85.84|   108.87|
|   310P3|ViT_base_patch16_384| 8 (最优性能)|ImageNet| top1: 86.01|top1: 85.84|   151.01|
|   310P3|ViT_base_patch32_224|            1|ImageNet| top1: 80.72|top1: 80.63|   431.89|
|   310P3|ViT_base_patch32_224|64 (最优性能)|ImageNet| top1: 80.72|top1: 80.63|  1679.63|
|   310P3|ViT_base_patch32_384|            1|ImageNet| top1: 83.35|top1: 83.29|   267.01|
|   310P3|ViT_base_patch32_384|32 (最优性能)|ImageNet| top1: 83.35|top1: 83.29|   596.55|

> 完整性能数据请查阅文件：`performances.md`