# FaceNet模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

`FaceNet` 是一个通用人脸识别系统：采用深度卷积神经网络（CNN）学习将图像映射到欧式空间。空间距离直接和图片相似度相关：同一个人的不同图像在空间距离很小，不同人的图像在空间中有较大的距离，可以用于人脸验证、识别和聚类。在800万人，2亿多张样本集训练后，FaceNet在LFW数据集上测试的准确率达到了99.63%，在YouTube Faces DB数据集上，准确率为95.12%。

- 参考实现：

  ```
  url=https://github.com/timesler/facenet-pytorch.git
  branch=master
  commit_id=555aa4bec20ca3e7c2ead14e7e39d5bbce203e4b
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  PNet/RNet/ONet/FaceNet:

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | image    | FLOAT32  | batchsize x 3 x H x W     | NCHW         |

- 输出数据

  PNet:

  | 输出数据 | 大小                  | 数据类型 | 数据排布格式 |
  | -------- | --------              | -------- | ------------ |
  | probs    | batchsize x 2 x H x W | FLOAT32  | NCHW         |
  | reg      | batchsize x 4 x H x W | FLOAT32  | NCHW         |


  RNet:

  | 输出数据 | 大小          | 数据类型 | 数据排布格式 |
  | -------- | --------      | -------- | ------------ |
  | regs     | batchsize x 4 | FLOAT32  | ND           |
  | cls      | batchsize x 2 | FLOAT32  | ND           |

  ONet:

  | 输出数据 | 大小           | 数据类型 | 数据排布格式 |
  | -------- | --------       | -------- | ------------ |
  | landmark | batchsize x 4  | FLOAT32  | ND           |
  | regs     | batchsize x 10 | FLOAT32  | ND           |
  | cls      | batchsize x 2  | FLOAT32  | ND           |

  FaceNet:

  | 输出数据 | 大小              | 数据类型 | 数据排布格式 |
  | -------- | --------          | -------- | ------------ |
  | class    | batchsize x class | FLOAT32  | ND           |

# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                            | 版本    | 环境准备指导                                                                                          |
| ------------------------------------------------------------    | ------- | ------------------------------------------------------------                                          |
| 固件与驱动                                                      | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                            | 6.0.RC1 | -                                                                                                     |
| Python                                                          | 3.7.5   | -                                                                                                     |
| PyTorch                                                         | 1.5.0+ | -                                                                                                     |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                                                                     |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git        # 克隆仓库的代码
   git checkout master         # 切换到对应分支
   cd ACL_PyTorch/contrib/cv/face/FaceNet              # 切换到模型的代码仓目录
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   git clone https://gitee.com/Ronnie_zheng/MagicONNX
   cd MagicONNX && git checkout master
   pip3 install .
   cd ..
   ```

3. 获取开源代码仓。
   在已下载的源码包根目录下，执行如下命令。

   ```
   git clone https://github.com/timesler/facenet-pytorch.git
   cd facenet-pytorch && git checkout 555aa4bec20ca3e7c2ead14e7e39d5bbce203e4b
   patch -p1 < ../models/mtcnn.patch
   cd ..
   cp ./facenet-pytorch/models/mtcnn.py models/
   ```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。

   本模型采用 [LFW(Labled Faces in Wild)](http://vis-www.cs.umass.edu/lfw/lfw.tgz) ，解压到 `./data` 目录下（如没有则需要自己创建）。另外，精度验证依赖`pairs.txt`文件，可以基于`FaceNet`仓预处理得到，也可以直接下载：[百度网盘](https://pan.baidu.com/s/11NCCf7Am7uV6eDL0dHvxUg)，提取码：iwyk，下载到`./data`目录下。

   数据目录结构请参考：

   ```
   data
   └── pairs.txt
   └── lfw
    ├── AJ_Cook
    ├── AJ_Lamas
    └── ...
   ```

2. 数据预处理。

   执行预处理脚本，生成数据集预处理后的bin文件:  `FACENet` 网络预处理依赖 `MTCNN` 模块推理，需要得到离线模型后才能进行预处理操作，故预处理操作和模型推理结合在一起。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      获取 [权重文件](https://pan.baidu.com/s/1hslY-6PZaqevSfZL3tWjwA)，提取码：1234 。主要得到相关pt文件：`pnet.pt/rnet.pt/onet.pt/Inception_facenet_vggface2.pt` ,将其拷贝到 `./weights` 文件夹下（如不存在则需要创建）。

   2. 导出onnx文件。

      1. 使用脚本导出onnx文件。

         运行MTCNN_pth2onnx.py/FaceNet_pth2onnx.py脚本。

         ```
         # MTCNN pth转换为ONNX
         python3 MTCNN_pth2onnx.py --model PNet --output_file ./weights/PNet_truncated.onnx
         python3 MTCNN_pth2onnx.py --model RNet --output_file ./weights/RNet_truncated.onnx
         python3 MTCNN_pth2onnx.py --model ONet --output_file ./weights/ONet_truncated.onnx
         ```
         - 参数说明：

           --model: 模型名。

           --output_file：输出onnx文件路径。

         获得文件PNet_truncated.onnx/RNet_truncated.onnx/ONet_truncated.onnx。

         ```
         # FaceNet pth转换为ONNX
         python3 FaceNet_pth2onnx.py --pretrain vggface2 --model ./weights/Inception_facenet_vggface2.pt --output_file ./weights/Inception_facenet_vggface2.onnx
         ```

         - 参数说明：

           --pretrain: 模型名。

           --model: 模型权重文件路径。

           --output_file：输出onnx文件路径。

         获得文件Inception_facenet_vggface2.onnx。

     2. 优化onnx。

        ```
        python utils/fix_prelu.py ./weights/PNet_truncated.onnx ./weights/PNet_truncated_fix.onnx
        python utils/fix_prelu.py ./weights/RNet_truncated.onnx ./weights/RNet_truncated_fix.onnx
        python utils/fix_prelu.py ./weights/ONet_truncated.onnx ./weights/ONet_truncated_fix.onnx
        python utils/fix_prelu.py ./weights/Inception_facenet_vggface2.onnx ./weights/Inception_facenet_vggface2_fix.onnx 
        python utils/fix_clip.py ./weights/Inception_facenet_vggface2_fix.onnx ./weights/Inception_facenet_vggface2_fix.onnx
        ```

        - 参数说明：第一个参数为原始onnx，第二个参数为优化后的onnx。

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
         # MTCNN：动态
         atc --framework=5 --model=./weights/PNet_truncated_fix.onnx --output=./weights/PNet_dynamic --input_format=NCHW --input_shape='image:[1~32,3,1~1500,1~1500]' --log=debug --soc_version=Ascend${chip_name} --log=error
         atc --framework=5 --model=./weights/RNet_truncated_fix.onnx --output=./weights/RNet_dynamic --input_format=NCHW --input_shape='image:[1~2000,3,24,24]' --log=debug --soc_version=Ascend${chip_name} --log=error
         atc --framework=5 --model=./weights/ONet_truncated_fix.onnx --output=./weights/ONet_dynamic --input_format=NCHW --input_shape='image:[1~1000,3,48,48]' --log=debug --soc_version=Ascend${chip_name} --log=error
         # FaceNet: 以bs1为例
         atc --framework=5 --model=./weights/Inception_facenet_vggface2_fix.onnx --output=./weights/Inception_facenet_vggface2_bs1 --input_format=NCHW --input_shape="image:1,3,160,160" --soc_version=Ascend${chip_name} --log=error
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --input\_shape\：动态模型输入数据的shape range。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成模型文件PNet_dynamic.om/RNet_dynamic.om/ONet_dynamic.om/Inception_facenet_vggface2_bs1.om。

2. 开始推理验证。

   1. 使用ais-bench工具进行推理。

      ais-bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)]

   2. 执行推理。

        ```
        # 基于MTCNN生成预处理数据
        python3 MTCNN_preprocess.py --model Pnet --data_dir ./data/lfw
        python3 MTCNN_preprocess.py --model Rnet --data_dir ./data/lfw
        python3 MTCNN_preprocess.py --model Onet --data_dir ./data/lfw
        ```
        -   参数说明：

             -   --model：模型名。
             -   --data_dir：数据集路径。

        推理后的输出默认在当前目录./data/output/split_bs1下, 用于下一阶段的预处理数据生成于./data/lfw_split_om_cropped_1

        ```
        # 基于MTCNN生成的数据进行预处理
        python3 FaceNet_preprocess.py --crop_dir ./data/lfw_split_om_cropped_1 --save_dir ./data/input/Facenet
        ```

        -   参数说明：

             -   --crop_dir：MTCNN推理得到的数据。
             -   --save_dir：预处理得到的数据保存路径。

        预处理数据生成于./data/input/Facenet,包括xb_results/yb_results两部分

        ```
        # facenet进行推理，以bs1为例
        python3 -m ais_bench --model ./weights/Inception_facenet_vggface2_bs1.om  --input ./data/input/Facenet/xb_results --output ./results --output_dirname bs1 --batchsize 1
        ```
        -   参数说明：

             -   --model：om文件路径。
             -   --input：输入文件。
             -   --output：输出目录。
             -   --output_dirname：保存目录名。
             -   --device：NPU设备编号。
             -   --outfmt: 输出数据格式。
             -   --batchsize：推理模型对应的batchsize。

        推理后的输出默认在当前目录results/bs1下。

   3.  精度验证。

      调用pointnet_postprocess.py脚本与数据集标签比对，获得Accuracy数据。

      ```
      # 以bs1为例
      mkdir -p results/bs1_pre
      python3 ./utils/batch_utils.py --batch_size 1 --data_root_path ./results/bs1 --save_root_path ./results/bs1_pre
      python3 FaceNet_postprocess.py  --ONet_output_dir ./data/output/split_bs1/onet.json --test_dir ./results/bs1_pre --crop_dir ./data/lfw_split_om_cropped_1
      ```

      -   参数说明：

        -  --ONet_output_dir: onet输出结果json配置文件路径。
        -  --test_dir: FaceNet推理得到结果路径。
        -  --crop_dir: MTCNN得到的crop结果路径。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

精度参考下列数据:

| 模型                       | pth精度    | 310精度    | 310P精度  |
| :------:                   | :------:   | :------:   | :------:  |
| Inception_facenet_vggface2 | ACC: 99.4% | ACC: 99.4% | ACC:99.2% |

推理性能：

| Model                      | Batch Size | 310(FPS/Card) | 310P3(FPS/Card) | 基准(FPS/Card) |
|----------------------------|------------|---------------|-----------------|----------------|
| Inception_facenet_vggface2 |          1 |        1693.7 |          1404.0 |        797.067 |
| Inception_facenet_vggface2 |          4 |        3034.6 |          3853.2 |        2473.96 |
| Inception_facenet_vggface2 |          8 |        4553.5 |          5811.6 |        3724.76 |
| Inception_facenet_vggface2 |         16 |        5336.4 |          7964.7 |        4727.60 |
| Inception_facenet_vggface2 |         32 |        3850.5 |          7645.9 |        5583.47 |
| Inception_facenet_vggface2 |         64 |             - |          7667.7 |         5876.6 |

