# vit-small 模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  ******

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

Vision Transformer是一个经典的图像分类网络。以前的cv领域虽然引入了transformer，但是都同时用到了cnn或者rnn。Vision Transformer直接使用纯transformer的结构并在图像识别上取得了不错的结果。本文档描述的是Vision Transformer中对配置为vit_small_patch16_224模型的Pytorch实现版本。

- 参考实现：

  ```
  url=branch=https://github.com/rwightman/pytorch-image-models.git
  commit_id=a41de1f666f9187e70845bbcf5b092f40acaf097
  model_name=vision_transformer
  ```

  通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output   | batchsize  x 1000 | FLOAT32  | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.7.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/rwightman/pytorch-image-models.git -b master
   cd pytorch-image-models/
   git reset --hard a41de1f666f9187e70845bbcf5b092f40acaf097
   patch -p1 < ../vit_small_patch16_224.patch
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用 `ImageNet` 官网的5万张验证集进行测试，可从 [ImageNet官网](http://www.image-net.org/) 获取 `val` 数据集与标签，存放在../dataset/ImageNet/下

   最终目录结构应为：

   ```bash
   ImageNet
   |-- ILSVRC2012_img_val/
   |-- ILSVRC2012_devkit_t12
       |-- val_label.txt
   ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行“vit_small_patch16_224_preprocess.py”脚本，完成预处理。

   ```
   python3 vit_small_patch16_224_preprocess.py ../dataset/ImageNet/ILSVRC2012_img_val  ./prep_dataset
   ```

   “../dataset/ImageNet/ILSVRC2012_img_val”：原始数据验证集（.jpeg）所在路径。

   “./prep_dataset”：输出的二进制文件（.bin）所在路径。

   每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“prep_dataset”二进制文件夹和“vit_prep_bin.info”。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       用户自行获取权重文件，并将其放到当前工作目录下，下面给出百度网盘链接以供下载：

       ```
       链接：https://pan.baidu.com/s/1UPGXtQwXH7aQYXyDFf5FAg 
       提取码：7rfl
       ```

   2. 导出onnx文件。

      使用“S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz”导出onnx文件。

         运行“vit_small_patch16_224_pth2onnx.py”脚本。

         ```
         python3 vit_small_patch16_224_pth2onnx.py ./S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz ./vit_small_patch16_224.onnx
         ```

         获得“vit_small_patch16_224.onnx”文件。


   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

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
         atc --framework=5 --model=vit_small_patch16_224.onnx --output=vit_small_patch16_224_bs1 --input_format=NCHW --input_shape="image:1,3,224,224" --log=error --soc_version=Ascend${chip_name} --enable_small_channel=1  --optypelist_for_implmode="Gelu" --op_select_implmode=high_performance
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --op\_select\_implmode：算子精度模式。

           运行成功后生成“vit_small_patch16_224_bs1.om”模型文件。

2. 开始推理验证。

   a.  使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]


   b.  执行推理。

      ```
      python3 ais_infer.py --model ./vit_small_patch16_224_bs1.om --input ./prep_dataset/ --batchsize 1 --output ./result --outfmt TXT
      ```

      -   参数说明：

           -   --model：模型类型。
           -   --input：预处理后的输入数据。
           -   --batchsize：om模型的batchsize。
           -   --output：推理结果存放目录。
		...

      推理后的输出在目录“./result/Timestam”下，Timestam为日期+时间的子文件夹,如 2022_08_11-10_37_29。

      >**说明：** 
      >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见。

   c.  精度验证。

      调用“vit_small_patch16_224_postprocess.py”脚本与数据集标签“val_label.txt”比对，可以获得Accuracy数据，结果保存在“result.json”中。

      ```
      rm ./result/2022_08_11-10_37_29/sumary.json
      python3 vit_small_patch16_224_postprocess.py ./result/2022_08_11-10_37_29/ ../dataset/ImageNet/ILSVRC2012_devkit_t12/val_label.txt ./ result.json
      ```

      “result/2022_08_11-10_37_29/”：为生成推理结果所在路径。
    
      “../dataset/ImageNet/ILSVRC2012_devkit_t12/val_label.txt”：为标签数据
    
      “result.json”：为生成结果文件


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

1.精度对比

| 模型           | 仓库pth精度 | 310离线推理精度 | 310P离线推理精度 |
| -------------- | ----------- | --------------- | --------------- |
| vit-small bs1  | top1:81.388 | top1:81.1 |  top1:81.37    |
| vit-small bs8 | top1:81.388 | top1:81.1  |  top1:81.37   |

2.性能对比
| Throughput | 310     | 310P    | T4     | 310P/310   | 310P/T4    |
| ---------- | ------- | ------- | ------ | ----------- | ----------- |
| bs1        | 203.054 | 435.2014 | 391.5258 | 2.14 | 1.11 |
| bs4        | 213.0816 | 771.9063 | 591.5261 | 3.62 | 1.30 |
| bs8        | 213.6788 | 1013.199 | 621.6682 | 4.74 | 1.63 |
| bs16       | 204.7552 | 913.2987 | 595.5638 | 4.46 | 1.53 |
| bs32       | 187.1448  | 778.4453  | 590.2469 | 4.15 | 1.31 |
| bs64       | 508.6636 | 730.0449 | 613.2265 | 1.19 | 1.19 |
|            |         |         |        |             |             |
| 最优batch  | 508.6636 | 1013.199 | 621.6682 | 1.99 | 1.63 |

