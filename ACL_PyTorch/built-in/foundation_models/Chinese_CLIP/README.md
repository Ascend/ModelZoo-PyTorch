# Chinese_CLIP模型推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

本项目为CLIP模型的中文版本，使用大规模中文数据进行训练（~2亿图文对），旨在帮助用户快速实现中文领域的图文特征&相似度计算、跨模态检索、零样本图片分类等任务。


- 参考实现：

  ```
  url=https://github.com/OFA-Sys/Chinese-CLIP.git
  commit_id=2c38d03557e50eadc72972b272cebf840dbc34ea
  model_name=clip_cn_vit-h-14
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 |   大小   |   数据类型      | 数据排布格式 |
  |----------|------------------| ------------------------- | ------------ |
  | image    | 1 x 3 x 224 x 224 | FLOAT32 | NCHW     |


- 输出数据

  | 输出数据 | 大小        | 数据类型 | 数据排布格式 |
  |-----------| -------- | -------- | ------------|
  | output  | 1 x 1024 | FLOAT32  | ND           |



# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------------ |---------| ------------------------------------------------------------ |
| 固件与驱动                                                   | 23.0.rc3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 7.0.RC1 | -                                                            |
| Python                                                       | 3.8.17   | -                                                            |
| PyTorch                                                      | 1.12.1   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/OFA-Sys/Chinese-CLIP.git
   cd Chinese-CLIP
   pip3 install -r requirements.txt
   cd ..
   ```

2.  安装依赖。

    1. 安装基础环境
    ```bash
    pip3 install -r requirements.txt
    ```
    说明：某些库如果通过此方式安装失败，可使用pip单独进行安装。
3. 设置环境变量。

   ```
   export PYTHONPATH=${PYTHONPATH}:`pwd`/Chinese-CLIP/cn_clip
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   数据路径：./Chinese-CLIP/examples/。

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行**preprocess_clip_pytorch.py**脚本，完成预处理
   ```
   python3 preprocess_clip_pytorch.py \
        --model_arch ViT-H-14 \
        --src_path ./Chinese-CLIP/examples/ \
        --npy_path ./npy_path 
   ```
   - 参数说明
        - --model_arch: 模型骨架
        - --src_path: 图片数据路径
        - --npy_path: 生成npy文件地址
   
   运行成功后，在npy_path目录下生成pokemon.npy文件



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
      ```
      mkdir models
      ```

      下载对应的[权重文件](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-h-14.pt)于models目录下

   2. 导出onnx文件。

      1. 使用Chinese-CLIP/cn_clip/deploy/pytorch_to_onnx.py导出onnx文件。

         a. 代码修改。
         ```
         patch -p1 < clip_pt2onnx.patch
         ```

         b. 运行pytorch_to_onnx.py脚本。
         ```
         python Chinese-CLIP/cn_clip/deploy/pytorch_to_onnx.py \
                --model-arch ViT-H-14 \
                --pytorch-ckpt-path ./models/clip_cn_vit-h-14.pt \
                --save-onnx-path ./models/vit-h-14 \
                --convert-vision

         ```
         - 参数说明
              - --model-arch: 模型骨架
              - --pytorch-ckpt-path: Pytorch模型ckpt路径
              - --save-onnx-path: 输出ONNX格式模型的路径
              - --convert-text: 指定是否转图像侧模型
         > **说明：**

         ```
         Finished PyTorch to ONNX conversion...
         >>> The text FP32 ONNX model is saved at ./models/vit-h-14.txt.fp32.onnx
         >>> The text FP16 ONNX model is saved at ./models/vit-h-14.txt.fp16.onnx with extra file ./models/vit-h-14.txt.fp16.onnx.extra_file
         >>> The vision FP32 ONNX model is saved at ./models/vit-h-14.img.fp32.onnx with extra file ./models/vit-h-14.img.fp32.onnx.extra_file
         >>> The vision FP16 ONNX model is saved at ./models/vit-h-14.img.fp16.onnx with extra file ./models/vit-h-14.img.fp16.onnx.extra_file
         ```

         我们使用vit-h-14.img.fp16.onnx文件位于models目录下.

   
   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：**
         该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend910B4 （自行替换）
         回显如下：
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       910B4     | OK              | 15.8         42                0    / 0              |
         | 0       0         | 0000:82:00.0    | 0            1074 / 21534                            |
         +===================+=================+======================================================+
         | 1       910B4     | OK              | 15.4         43                0    / 0              |
         | 0       1         | 0000:89:00.0    | 0            1070 / 21534                            |
         +===================+=================+======================================================+
         ```

      3. 执行ATC命令。
         ```
         atc --framework=5 \
            --model=models/vit-h-14.img.fp16.onnx \
            --output=models/vit-h-14.img.fp16 \
            --input_format=NCHW \
            --input_shape="image:1,3,224,224" \
            --soc_version=Ascend${chip_name} \
            --log=error
         ```
         - 参数说明：
            - --model：为ONNX模型文件。
            - --framework：5代表ONNX模型。
            - --output：输出的OM模型。
            - --input_format：输入数据的格式。
            - --input_shape：输入数据的shape。
            - --log：日志级别。
            - --soc_version：处理器型号。
         
         运行成功后在models目录下生成**vit-h-14.img.fp16.om**模型文件。



2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。


   b.  执行推理。

      ```
      python3 -m ais_bench --model ./models/vit-h-14.img.fp16.om --input ./npy_path --output ./ --outfmt NPY --output_dirname dst

      ```

      -   参数说明：

           -   model：需要推理om模型的路径。
           -   input：模型需要的输入bin文件夹路径。
           -   output：推理结果输出路径。
           -   outfmt：输出数据的格式。
           -   output_dirname:推理结果输出子文件夹。


   c.  精度验证。

      1. 在线模型推理

      调用clip_onnx_infer.py脚本。
      ```
      python3 clip_onnx_infer.py \
             --model_arch ViT-H-14 \
             --model_path ./models/vit-h-14.img.fp16.onnx \
             --src_path ./Chinese-CLIP/examples \
             --npy_path ./onnx_npy_path 
      ```
      - 参数说明
        - --model_arch: 模型骨架
        - --model_path: onnx模型路径
        - --src_path: 图片数据路径
        - --npy_path: onnx模型推理结果npy文件地址
   
      运行成功后，在onnx_npy_path目录下生成pokemon.npy文件

      2. 精度验证

      ```
      python3 compute_cosine_similarity.py \
             --onnx_npy_path ./onnx_npy_path \
             --om_npy_path ./dst
      ```
      - 参数说明

        - --onnx_npy_path：在线推理结果npy文件路径
        - --om_npy_path：离线推理结果npy文件路径
      
      精度验证结果：
      |  余弦相似度  |
      |-------------|
      |0.9999162032670914 | 
 
   
   d.  性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3 -m ais_bench --model ./models/vit-h-14.img.fp16.om --loop 100 --batchsize 1

      ```
      - 参数说明
        - --model: om模型
        - --loop: 循环次数
        - --batchsize: 模型batch size


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号  | Batch Size | 迭代次数  | 性能    | 
|----------|------------|----------|-------|
|  910B4  |       1       | 100 | 72.426fps |
