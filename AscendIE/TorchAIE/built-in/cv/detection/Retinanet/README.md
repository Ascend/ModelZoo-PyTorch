# RetinaNet模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

论文提出了一个简单、灵活、通用的损失函数Focal loss，用于解决单阶段目标检测网络检测精度不如双阶段网络的问题。这个损失函数是针对了难易样本训练和正负样本失衡所提出的，使得单阶段网络在运行快速的情况下，获得与双阶段检测网络相当的检测精度。此外作者还提出了一个Retinanet用于检验网络的有效性，其中使用Resnet和FPN用于提取多尺度的特征。

- 参考实现：

  ```shell
  url=https://github.com/facebookresearch/detectron2
  commit_id=60fd4885d7cfd52d4267d1da9ebb6b2b9a3fc937
  code_path=detectron2
  model_name=detectron2
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小              | 数据排布格式 |
  | -------- | -------- | ----------------- | ------------ |
  | input    | FP32     | 1 x 3 x 1344 x 1344 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小    | 数据排布格式 |
  | -------- | -------- | ------- | ------------ |
  | boxes    | FLOAT32  | 4 x 100 | ND           |
  | labels   | INT64    | 1 x 100 | ND           |
  | scores   | FLOAT32  | 100 | ND           |

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本      | 环境准备指导                                                 |
  |---------| ------- | ------------------------------------------------------------ |
  | 固件与驱动 | 23.0.rc1  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN | 7.0.RC1.alpha003 | - |
  | Python | 3.9.11 | - |
  | PyTorch | 2.0.1 | - |
  | Torch_AIE | 6.3.rc2 | - | - |                                                                                          |

- 安装依赖

   ```
   pip install -r requirements.txt
   ```

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

由于该模型使用Torch_AIE推理，要求torch版本为2.0.1，而detectron2不支持该版本torchscript模型导出。所以数据准备，模型导出，需要安装torch==1.10.1, torchvision==0.11.2，而推理时使用torch==2.0.1, torchvision==0.15.2。 

## 准备数据集<a name="section183221994411"></a>


1. 数据准备。

   参考[Retinanet昇腾离线模型推理](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Retinanet), 获取预处理后的数据`val2017_bin`。注意该步骤和后续模型导出torch版本为1.10.1，而不是离线模型推理使用的1.8.0。

## 模型推理<a name="section741711594517"></a>

1. 源码获取。

   同样参考[Retinanet昇腾离线模型推理](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Retinanet)，在获取源码时引入额外修改Retina_aie.diff：
   ```shell
      git clone https://github.com/facebookresearch/detectron2 -b main
      cd detectron2
      git reset --hard 60fd4885d7cfd52d4267d1da9ebb6b2b9a3fc937
      patch -p1 < ../Retinanet.diff
      patch -p1 < ../Retina_aie.diff
      pip install -e .
      cd -
   ```

2. Torchscript算子编译
   参考[Maskrcnn-mmdet 模型推理指导](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/AscendIE/TorchAIE/built-in/cv/semantic-segmentation/Maskrcnn-mmdet)模型序列化第二步，使用相同的mmdet_ops源码编译自定义算子，生成动态库`./mmdet_ops/build/libmmdet_ops.so`。


3. 导出torchscript文件。

   ```shell
   python detectron2/tools/deploy/export_model.py \
         --config-file detectron2/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml \
         --output ./ \
         --export-method tracing \
         --format torchscript MODEL.WEIGHTS RetinaNet-detectron2.pkl MODEL.DEVICE cpu
   ```

   获得model.ts文件。

4. 执行推理。

   ```
   python3 inference.py
   ```
   推理结果保存在result文件夹中。

5. 精度验证。

   同样参考[Retinanet昇腾离线模型推理](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Retinanet) 进行精度验证，注意使用推理时的torch版本即可。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

1. 精度对比

    | Model     | batchsize | Accuracy    | 开源仓精度  |
    | --------- | --------- | ----------- | ----------- |
    | Retinanet | 1         | map = 38.3% | map = 38.6% |

2. 性能对比

    | batchsize  | 310P3 性能 |
    | --------- | ---------- |
    | 1         | 19.9       |