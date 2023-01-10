#  Wseg 模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

------

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

wseg作为语义分割方法与之前工作相比的特点在于：1. 不需要先检测再分割；2. 不需要有像素级别的标注信息，只需要图像层次的类别信息就能训练。

- 参考论文：

  *[Araslanov N, Roth S. Single-stage semantic segmentation from image labels](https://arxiv.org/abs/2005.08104)*

- 参考实现：

  ```
  url=https://github.com/visinf/1-stage-wseg
  commit_id=cfe5784f9905d656e0f15fba0e6eb76a3731d80f
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                       | 数据排布格式 |
  | -------- | -------- | -------------------------- | ------------ |
  | image    | RGB_FP32 | batchsize x 3 x 1020 x1020 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- | -------- | --------------------------- | ------------ |
  | mask     | RGB_FP32 | batchsize x 3 x 2040 x 2040 | NCHW         |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1** 版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FModelZoo%2Fpytorchframework%2Fpies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.12.1  | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码，放于自建目录 `<project>` 下

2. 获取开源模型代码，与第一步源码放于同级目录 `<project>` 下。

   ```
   git clone https://github.com/visinf/1-stage-wseg -b master   
   cd 1-stage-wseg
   git reset cfe5784f9905d656e0f15fba0e6eb76a3731d80f --hard
   cd ..
   mv 1-stage-wseg wseg
   ```
   
3. 安装必要依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   下载VOC数据集[Training/Validation (2GB .tar file)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)。将文件名改为voc，并将其放于代码仓中的以下路径： `<project>/data/`，得到如下目录结构

   ```
   <project>/data/voc/VOCdevkit/VOC2012
   ├──SegmentationObject
   ├──SegmentationClass
   ├──JPEGImages
   ├──Annotations
   ├──ImageSets
   │   ├── Segmentation
   │   ├── Main
   │   ├── Layout
   ```

2. 数据预处理，将原始数据集转换为模型的输入数据。

   执行 wseg_preprocess.py 脚本，完成数据预处理。

   ```
   python3 pth2onnx.py ./data ./wseg/data/val_voc.txt ${prep_data}
   ```

   参数说明：

   - --参数1：原数据集路径。
   - --参数2：验证集文件列表。
   - --参数3：为生成数据集文件的保存路径。
   

运行成功后，在当前目录生成预处理后的数据集文件。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件 。

      a. 获取经过预训练的基础网络权重文件并且放在代码仓的以下路径中：`<project>/models/weights/`.

      |   Backbone   |                             Link                             |
      | :----------: | :----------------------------------------------------------: |
      | WideResNet38 | [ilsvrc-cls_rna-a1_cls1000_ep-0001.pth (402M)](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-araslanov-1-stage-wseg/models/ilsvrc-cls_rna-a1_cls1000_ep-0001.pth) |

      b. 获取功能网络权重（作者提供的pth模型）并放置于代码仓的以下路径中：（初始代码仓无snapshots文件夹，需要自己新建路径）`<project>/snapshots/`

      |   Backbone   |                                                         Link |
      | :----------: | -----------------------------------------------------------: |
      | WideResNet38 | [model_enc_e020Xs0.928.pth (527M)](https://download.visinf.tu-darmstadt.de/data/2020-cvpr-araslanov-1-stage-wseg/models/model_enc_e020Xs0.928.pth) |

      c. 移动上述两个权重文件到代码仓指定位置，以待加载使用

      ```
      mkdir ./models/weights
      mv ilsvrc-cls_rna-a1_cls1000_ep-0001.pth ./models/weights
      mkdir ./snapshots
      mv model_enc_e020Xs0.928.pth ./snapshots
      ```

   2. 导出onnx文件。

      1. 使用wseg_pth2onnx.py导出动态batch的onnx文件`wideresnet_dybs.onnx`。

         ```
         python3 wseg_pth2onnx.py ./wseg/configs/voc_resnet38.yaml ./snapshots/model_enc_e020Xs0.928.pth wideresnet_dybs.onnx
         ```

         参数说明：

         - --参数1：模型配置文件。
         - --参数2：模型权重文件。
         - --参数3：生成的onnx文件名称。
         
      2. 使用fix_onnx.py得到修改后的onnx文件。
      
         ```
         python3 fix_softmax_transpose.py ./wideresnet_dybs.onnx ./wideresnet_dybs_fix.onnx
         ```
      
         参数说明：
      
         - --参数1：原onnx文件。
         - --参数2：修改后的onnx文件
      
   3. 使用ATC工具将ONNX模型转OM模型。
   
      1. 配置环境变量。
   
         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```
   
      2. 执行命令查看芯片名称（${chip_name}）。
   
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
   
   4. 执行ATC命令。
   
      ```
      # bs = [1, 4, 8, 16]
      atc --model=./wideresnet_dybs_fix.onnx --framework=5 --output=wideresnet_bs${bs} --input_format=NCHW --input_shape="image:${bs},3,1024,1024" --log=error --soc_version=Ascend${chip_name}
      ```
      
      运行成功后生成wseg_bs${bs}.om模型文件。
      
      参数说明：
      
      - --model：为ONNX模型文件。
      - --framework：5代表ONNX模型。
      - --output：输出的OM模型。
      - --input_format：输入数据的格式。
      - --input_shape：输入数据的shape。
      - --log：日志级别。
      - --soc_version：处理器型号。
   
2. 开始推理验证。

   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

      ```
      python3 -m ais_bench --model=wideresnet_bs${bs}.om --batchsize=${bs} \
      --input ${prep_data} --output result --output_dirname result_bs${bs} --outfmt BIN
      ```
      
      参数说明：
      
      - --model：om模型路径。
      - --batchsize：批次大小。
      - --input：输入数据所在路径。
      - --output：推理结果输出路径。
      - --output_dirname：推理结果输出子文件夹。
      - --outfmt：推理结果输出格式
   
3. 精度验证。

   1. 调用wseg_postprocess.py脚本，在指定路径/output/output_bs${bs}生成后处理得到的结果文件。

      ```
      python3 wseg_postprocess.py ./data ./wseg/data/val_voc.txt ./result/result_bs${bs}/ ./output/output_bs${bs}/
      ```

      参数说明：

      - --参数1：原数据集路径。
      - --参数2：验证集文件列表。
      - --参数3：推理结果所在路径。
      - --参数4：后处理后生成的图片。

   2. 调用开源仓的精度验证脚本，计算精度。

      ```
      python3 ./wseg/eval_seg.py --data ./data --filelist wseg/data/val_voc.txt --masks output/output_bs${bs}/crf
      ```

      参数说明：

      - --data：原数据集路径。
      - --filelist：验证集文件列表。
      - --masks：后处理得到的结果文件。

4. 可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

   ```
   python3 -m ais_bench --model=wideresnet_bs${bs}.om --loop=50 --batchsize=${bs}
   ```

   参数说明：

   - --model：om模型路径。
   - --loop：纯推理循环次数。
   - --batchsize：批次大小。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，Wseg模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集  | 开源精度（IoU）                                 | 参考精度（IoU） |
| ----------- | ---------- | ------- | ----------------------------------------------- | --------------- |
| Ascend310P3 | 1          | VOC2012 | [62.7%](https://github.com/visinf/1-stage-wseg) | 62.74%          |

| 芯片型号    | Batch Size | 参考性能（FPS） |
| ----------- | ---------- | --------------- |
| Ascend310P3 | 1          | 5.26            |
| Ascend310P3 | 4          | 3.68            |
| Ascend310P3 | 8          | 2.29            |
| Ascend310P3 | 16         | 2.28            |

