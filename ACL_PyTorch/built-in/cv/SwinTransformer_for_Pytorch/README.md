# SwinTransformer模型-推理指导

## 概述

`Swin Transformer`是在`Vision Transformer`的基础上使用滑动窗口（shifted windows, SW）进行改造而来。它将`Vision Transformer`中固定大小的采样快按照层次分成不同大小的块（Windows），每一个块之间的信息并不共通、独立运算从而大大提高了计算效率。

+ 参考实现：

  ```shell
  url=https://github.com/rwightman/pytorch-image-models
  mode_name=SwinTransformer
  ```

  通过Git获取对应的commit_id的代码方法如下：

  ```shell
  git clone https://gitee.com/ascend/ModelZoo-PyTorch
  cd ModelZoo-PyTorch
  git checkout master
  cd ACL_PyTorch/built-in/cv/SwinTransformer_for_Pytorch
  ```


## 输入输出数据

+ 输入数据

  | 输入数据 | 数据类型 | 大小                           | 数据排布格式 |
  |----------|----------|--------------------------------|--------------|
  | image    | FP16     | batchsize x 3 x height x width | NCHW         |

+ 输出数据

  | 输出数据 | 数据类型 | 大小                  | 数据排布格式 |
  |----------|----------|-----------------------|--------------|
  | out      | FP16     | batchsize x num_class | ND           |


## 推理环境准备

+ 表1 版本配套表
  
  | 配套                                                          | 版本    | 环境准备指导                                                                                          |
  |---------------------------------------------------------------|---------|-------------------------------------------------------------------------------------------------------|
  | 固件与驱动                                                    | 5.1.RC2 | [Pytorch框架推理环境准别](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                          | 5.1.RC2 |                                                                                                       |
  | Python                                                        | 3.7+    |                                                                                                       |
  | PyTorch                                                       | 1.8.0+  |                                                                                                       |
  | 说明：Altlas 300I Duo推理卡请以CANN版本选择实际固件于驱动版本 | \       | \                                                                                                     |


## 快速上手

1. 安装依赖（建议按照自己需求安装）

   基础依赖：

   ```shell
   pip3 install -r requirement.txt
   ```

   安装改图工具依赖：

   ```shell
   git clone https://gitee.com/Ronnie_zheng/MagicONNX.git MagicONNX
   cd MagicONNX && git checkout cb071bb62f34bfae405af52063d7a2a4b101358a
   pip3 install . && cd ..
   ```

## 准备数据集

1. 获取原始数据集

   本模型使用ImageNet 50000张图片的验证集，请前往[ImageNet官网](https://image-net.org/download.php)下载数据集:

    ```
    ├── ImageNet
    |   ├── val
    |   |    ├── ILSVRC2012_val_00000001.JPEG
    │   |    ├── ILSVRC2012_val_00000002.JPEG
    │   |    ├── ......
    |   ├── val_label.txt
    ```

2. 数据预处理

   ```shell
   python3 preprocess.py --img_size 384 --input_dir /opt/npu/imagenet/val --out_dir ./preprocessed_data
   ```

## 模型推理

1. 模型转换

   获取权重文件：[swin_base_patch4_window12_384_22kto1k.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth)，其他模型权重下载地址可参考：[Swin-Transformer(microsoft)](https://github.com/microsoft/Swin-Transformer)

2. 导出onnx文件:

   ```shell
   # bs:[1, 4, 8, 16, 32, 64]
   python3 pth2onnx.py -input_path swin_base_patch4_window12_384_22kto1k.pth --out_path models/onnx/swin_base_patch4_window12_384_bs${bs}.onnx --output_type=FP16 --model_name swin_base_patch4_window12_384 --batch_size ${bs}
   python3 -m onnxsim models/onnx/swin_base_patch4_window12_384_bs${bs}.onnx models/onnx/swin_base_patch4_window12_384_bs${bs}.onnx
   python3 opt_onnx.py -i models/onnx/swin_base_patch4_window12_384_bs${bs}.onnx -o models/onnx/swin_base_patch4_window12_384_bs${bs}.onnx
   ```

3. 使用ATC工具将ONNX模型转OM模型

   1. 配置环境变量

      ```shell
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      ```

   2. 执行命令查看芯片名称(${chip_name})

      ```shell
      npu-smi info  # 通过 Name Device列获取${chip_name}
      ```

   3. 执行ATC命令

      ```shell
      # bs:[1, 4, 8, 16, 32, 64]
      # 以bs8模型为例
      atc --model=models/onnx/swin_base_patch4_window12_384_bs8.onnx --framework=5 --output=models/om/swin_base_patch4_window12_384_bs8 --input_format=NCHW --log=debug --soc_version=${chip_name} --output_type=FP16 --input_fp16_nodes="image" --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
   ```
      
   + 参数说明：
      
        + --model ：为ONNX模型文件
        + --framework：5代表ONNX模型
        + --output：输出的OM模型
        + --input_format：输入数据的格式
        + --input_shape：输入数据的shape
        + --log：日志级别
        + --soc_version：处理器型号
        + --optypelist_for_implmode：需要选择implmode的算子
        + --op_select_implmode：特定算子的实现方式
        
        运行成功后生成OM模型文件。

4. 使用ais-infer工具进行推理

   安装过程可参考：[ais_infer](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)

5. 执行推理

   ```shell
   # 以bs8模型推理为例
   python3 ais_infer.py --model models/om/swin_base_patch4_window12_384_bs8.om --input ./preprocessed_data --output ./output_data/bs8 --device 1
   ```

6. 精度验证

   ```shell
   # 以bs8模型推理为例
   python3 postprocess.py --input_dir ./output/bs8/${timestamp} --label_path ./val_label.txt --save_path ./result_bs8.json
   ```

## 模型推理性能&精度

调用ACL接口推理计算，性能参考下列数据：

精度：

| 模型            | Pth精度              | NPU离线推理精度      |
|:---------------:|:--------------------:|:--------------------:|
| SwinTransformer | Top1 Acc: 86.4%;Top5 Acc: 98.0%| Top1 Acc: 86.4%;Top5 Acc: 98.0%|

性能：

| 模型            | BatchSize | NPU性能 | 基准性能  |
|:---------------:|:---------:|:-------:|:---------:|
| SwinTransformer | 1         | 104 fps | 217 fps |
| SwinTransformer | 4         | 105 fps | 332 fps |
| SwinTransformer | 8         | 140 fps | 339 fps |
| SwinTransformer | 16        | 124 fps | 351 fps |
| SwinTransformer | 32        | 126 fps | 357 fps |
| SwinTransformer | 64        | 126 fps | 363 fps |
