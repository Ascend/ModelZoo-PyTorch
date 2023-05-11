# 交付件基本信息

**发布者（Publisher）**：Huawei

**应用领域（Application Domain）**：Image Classification

**版本（Version）**：1.2

**修改时间（Modified）**：2021.09.10

**大小（Size）**：233MB \(pyth\)/117 MB \(onnx\)/83MB \(om\)

**框架（Framework）**：PyTorch (1.5.0)

**模型格式（Model Format）**：pyth/onnx/om

**精度（Precision）**：O2(训练)、FP16（推理）

**处理器（Processor）**：昇腾910/昇腾310

**应用级别（Categories）**：Released

**描述（Description）：**基于Pytorch框架的 EfficientNet-B5目标检测网络模型训练并保存模型，通过ATC工具转换，可在昇腾AI设备上运行

# 概述

## 简述

描述要点（key）：2019年，谷歌新出EfficientNet，这个网络非常的有效率，它利用更少的参数量（关系到训练、速度）得到最好的识别度（学到更重要的特点）。EfficientNet模型具有很独特的特点，经典的神经网络特点如下：

1、利用残差神经网络增大神经网络的深度，通过更深的神经网络实现特征提取。
2、改变每一层提取的特征层数，实现更多层的特征提取，得到更多的特征，提升宽度。
3、通过增大输入图片的分辨率也可以使得网络可以学习与表达的东西更加丰富，有利于提高精确度。

- 参考论文：[Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 1-9.](http://arxiv.org/abs/1409.4842)

- 参考实现：https://github.com/facebookresearch/pycls

通过Git获取对应commit_id的代码方法如下：

```
git clone {repository_url}     # 克隆仓库的代码
cd {repository_name}           # 切换到模型的代码仓目录
git checkout  {branch}         # 切换到对应分支
git reset --hard ｛commit_id｝  # 代码设置到对应的commit_id
cd ｛code_path｝                # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```

## 默认配置

- 训练数据集预处理：

  图像的输入尺寸为488*488

  随机裁剪图像尺寸

  随机水平翻转图像

  根据ImageNet数据集通用的平均值和标准偏差对输入图像进行归一化

- 测试数据集预处理

  图像的输入尺寸为488*488（将图像最小边缩放到522，同时保持宽高比，然后在中心裁剪图像）

  根据ImageNet数据集通用的平均值和标准偏差对输入图像进行归一化

- 训练超参

  BATCH_SIZE: 12

  BASE_LR: 0.0375

  MAX_EPOCH: 40

  MOMENTUM: 0.9

  WEIGHT_DECAY: 1e-5

  LR_POLICY: cos

## 准备工作

### 推理环境准备

- 硬件环境、开发环境和运行环境准备请参见[《CANN 软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-upgrade)》。

- 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/home)获取镜像。

  当前模型支持的镜像列表如下表所示。

  **表 1**  镜像列表

  <table><thead align="left"><tr id="zh-cn_topic_0000001205858411_row0190152218319"><th class="cellrowborder" valign="top" width="55.00000000000001%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001205858411_p1419132211315"><a name="zh-cn_topic_0000001205858411_p1419132211315"></a><a name="zh-cn_topic_0000001205858411_p1419132211315"></a>镜像名称</p>
  </th>
  <th class="cellrowborder" valign="top" width="20%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001205858411_p75071327115313"><a name="zh-cn_topic_0000001205858411_p75071327115313"></a><a name="zh-cn_topic_0000001205858411_p75071327115313"></a>镜像版本</p>
  </th>
  <th class="cellrowborder" valign="top" width="25%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001205858411_p1024411406234"><a name="zh-cn_topic_0000001205858411_p1024411406234"></a><a name="zh-cn_topic_0000001205858411_p1024411406234"></a>配套CANN版本</p>
  </th>
  </tr>
  </thead>
  <tbody><tr id="zh-cn_topic_0000001205858411_row71915221134"><td class="cellrowborder" valign="top" width="55.00000000000001%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001205858411_p58911145153514"><a name="zh-cn_topic_0000001205858411_p58911145153514"></a><a name="zh-cn_topic_0000001205858411_p58911145153514"></a>ARM/x86架构：<a href="https://ascendhub.huawei.com/#/detail/infer-modelzoo" target="_blank" rel="noopener noreferrer">infer-modelzoo</a></p>
  </td>
  <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001205858411_p14648161414516"><a name="zh-cn_topic_0000001205858411_p14648161414516"></a><a name="zh-cn_topic_0000001205858411_p14648161414516"></a>21.0.2</p>
  </td>
  <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001205858411_p1264815147514"><a name="zh-cn_topic_0000001205858411_p1264815147514"></a><a name="zh-cn_topic_0000001205858411_p1264815147514"></a><a href="https://www.hiascend.com/software/cann/commercial" target="_blank" rel="noopener noreferrer">5.0.2</a></p>
  </td>
  </tr>
  </tbody>
  </table>

### 源码介绍

介绍训练脚本的源码+重要目录介绍

示例：

```
/home/data/ics_yu/txyworkspace/PR/EfficientNet-B5/modelzoo/contrib/PyTorch/Research/cv/image_classification/EfficientNet-B5_ID1621_for_PyTorch     #此代码路径需根据实际路径进行替换
├── infer                # MindX高性能预训练模型新增  
│   └── README.md        # 离线推理文档
│   ├── convert          # 转换om模型命令，AIPP
│   │   ├──efficientnetb5_aipp.cfg
│   │   └──convert_om.sh
│   ├── data             # 包括模型文件、模型输入数据集、模型相关配置文件（如label、SDK的pipeline）
│   │   ├── input
│   │   │   └──ILSVRC2012_val_00000001.JPEG 
│   │   └── config
│   │   │   ├──efficientnetb5_sdk_infer.cfg
│   │   │   └──efficientnetb5.pipeline
			└──imagenet1000_clsidx_to_labels.names
│   ├── mxbase           # 基于mxbase推理
│   │   ├── src
│   │   │   ├── EfficientNetB5Classify.cpp
│   │   │   ├── EfficientNetB5Classify.h
│   │   │   └── main.cpp
│   │   ├── CMakeLists.txt
│   │   └── build.sh
│   └── sdk               # 基于sdk.run包推理；如果是C++实现，存放路径一样
│   │   ├── main.py
│   │   └── run.sh
│   └── util              
│   │   └──classification_task_metric.py	# 精度验证脚本
│   └──docker_start_infer.sh     # 启动容器脚本
```

## 推理

-   **[准备推理数据](#准备推理数据.md)**  
-   **[模型转换](#模型转换.md)**  
-   **[mxBase推理](#mxBase推理.md)**  
-   **[MindX SDK推理](#MindX-SDK推理.md)**  

### 准备推理数据

准备推理数据

1. 下载源码包。（例如：单击“下载模型脚本”和“下载模型”，下载所需软件包。）

2. 将源码上传至推理服务器任意目录并解压（如：“/home/HwHiAiUser“）。

3. 准备数据。

   由于后续推理均在容器中进行，因此需要把用于推理的图片、数据集、模型文件、代码等均放在同一数据路径中，后续示例将以“/home/HwHiAiUser“为例。

```
├── infer                 # MindX高性能预训练模型新增  
│   └── README.md         # 离线推理文档
│   ├── convert           # 转换om模型命令，AIPP
│   │   ├──efficientnetb5_aipp.cfg
│   │   └──convert_om.sh
│   ├── data              # 包括模型文件、模型输入数据集、模型相关配置文件（如label、SDK的pipeline）
│   │   ├── input
│   │   ├── model         # air、om模型文件
│   │   ├── image_val     # 推理所需的数据集
│   │   └── config        # 推理所需的配置文件
│   ├── mxbase            # 基于mxbase推理
│   └── sdk               # 基于sdk.run包推理；如果是C++实现，存放路径一样
│   └── util             
│   │   └──classification_task_metric.py # 精度验证脚本
│   └──docker_start_infer.sh     # 启动容器脚本
```

AIR模型可通过“模型训练”后转换生成或通过“下载模型”获取。

将Imagenet数据集val中的所有图片放到“infer/data/image_val”目录下。

​	4.启动容器。

​	进入“infer“目录，执行以下命令，启动容器。

​	bash docker\_start\_infer.sh   docker\_image:tag   model\_dir

​	**表 2**  参数说明

<table><thead align="left"><tr id="row16122113320259"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p16122163382512"><a name="p16122163382512"></a><a name="p16122163382512"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p8122103342518"><a name="p8122103342518"></a><a name="p8122103342518"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row11225332251"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p712210339252"><a name="p712210339252"></a><a name="p712210339252"></a><em id="i121225338257"><a name="i121225338257"></a><a name="i121225338257"></a>docker_image</em></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p0122733152514"><a name="p0122733152514"></a><a name="p0122733152514"></a>推理镜像名称，根据实际写入。</p>
</td>
</tr>
<tr id="row052611279127"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p2526192714127"><a name="p2526192714127"></a><a name="p2526192714127"></a><em id="i12120733191212"><a name="i12120733191212"></a><a name="i12120733191212"></a>tag</em></p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p16526142731219"><a name="p16526142731219"></a><a name="p16526142731219"></a>镜像tag，请根据实际配置，如：21.0.2。</p>
</td>
</tr>
<tr id="row5835194195611"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p59018537424"><a name="p59018537424"></a><a name="p59018537424"></a>model_dir</p>
</td>
<td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1390135374214"><a name="p1390135374214"></a><a name="p1390135374214"></a>推理代码路径。</p>
</td>
</tr>
</tbody>
</table>

启动容器时会将推理芯片和数据路径挂载到容器中。可根据需要通过修改**docker\_start\_infer.sh**的device来指定挂载的推理芯片。

```
docker run -it \
  --device=/dev/davinci0 \         # 可根据需要修改挂载的npu设备
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v ${data_path}:${data_path} \
  ${docker_image} \
  /bin/bash
```

![](images/note_3.0-zh-cn.png)

MindX SDK开发套件（mxManufacture）已安装在基础镜像中，安装路径：“/usr/local/sdk\_home“。


   ## 模型转换

   1. 准备模型文件。

      将EfficientNet-B5的onnx模型文件放在/home/HwHiAiUser/EfficientNet-B5_ID1621_for_PyTorch/infer/convert/ 路径下

   2. 模型转换。

      输出om模型文件的convert_om.sh脚本在/home/HwHiAiUser/EfficientNet-B5_ID1621_for_PyTorch/infer/路径下供后续推理使用，脚本中部分内容的显示如下所示：

      ```
      atc --input_format=NCHW \
          --framework=5 \
          --model="${input_onnx_path}" \
          --input_shape="image:1,3,488,488"  \
          --output="${output_om_path}" \
          --insert_op_conf="${aipp_cfg_file}" \
          --enable_small_channel=1 \
          --log=error \
          --soc_version=Ascend310 \
      ```

​      转换命令如下。               

```
bash atc.sh  model_path  efficinetb5_aipp.cfg  output_model_name      
```

​      **表 1**  参数说明       

| 参数                 | 说明                                            |
| -------------------- | ----------------------------------------------- |
| model_path           | AIR文件路径                                     |
| efficinetb5_aipp.cfg | efficinetb5模型的cfg文件                        |
| output_model_name    | 生成的OM文件名，转换脚本会在此基础上添加.om后缀 |

   ## mxBase推理

#### 前提条件

 在容器内用mxBase进行推理。

首先进入/home/HwHiAiUser/EfficientNet-B5_ID1621_for_PyTorch/infer/路径下（文件所在路径需根据实际情况进行修改）。

1.修改配置文件。

根据实际情况修改initParam.labelPath = "../sdk/models/imagenet1000_clsidx_to_labels.names"；initParam.modelPath = "../sdk/models/xception_pt_pytorch.om"等。

```
...
	InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = "../sdk/models/imagenet1000_clsidx_to_labels.names";
    initParam.topk = 5;
    initParam.softmax = true;
    initParam.checkTensor = true;
    initParam.modelPath = "../sdk/models/xception_pt_pytorch.om";
    auto xception = std::make_shared<EfficientNetB5Classify>();
    APP_ERROR ret = xception->Init(initParam);
...
```

2.编译工程。

目前mxBase推理仅实现了基于opencv方式推理。

```
cd mxbase
bash build.sh
```

3.运行推理服务。

在当前路径下执行     

```
./build/efficientnetb5   /home/data/mindx/dataset/imagenet_val/
```

其中，/home/data/mindx/dataset/imagenet_val/为做mxbase推理时，数据集所在的位置

4.观察结果。

```
0910 08:52:24.887820  3967 ModelInferenceProcessor.cpp:33] Begin to ModelInferenceProcessor init
I0910 08:52:25.427723  3967 ModelInferenceProcessor.cpp:77] End to ModelInferenceProcessor init
I0910 08:52:25.427942  3967 Resnet50PostProcess.cpp:35] Start to Init Resnet50PostProcess.
I0910 08:52:25.427955  3967 PostProcessBase.cpp:78] Start to LoadConfigDataAndLabelMap in  PostProcessBase.
I0910 08:52:25.437688  3967 Resnet50PostProcess.cpp:44] End to Init Resnet50PostProcess.
I0910 08:52:25.446209  3967 EfficientNetB5Classify.cpp:130] images crop_x1: 16, crop_x: 489, crop_y1: 16, crop_y: 489
I0910 08:52:25.488122  3967 Resnet50PostProcess.cpp:73] Start to Process Resnet50PostProcess.
I0910 08:52:25.488847  3967 Resnet50PostProcess.cpp:120] End to Process Resnet50PostProcess.
I0910 08:52:25.488863  3967 EfficientNetB5Classify.cpp:241] image path: /home/data/mindx/dataset/imagenet_val/ILSVRC2012_val_00047110.JPEG
I0910 08:52:25.488881  3967 EfficientNetB5Classify.cpp:243] file path for saving result: result/ILSVRC2012_val_00047110_1.txt
I0910 08:52:25.488991  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top1 className:window screen confidence:11.5703 classIndex:904
I0910 08:52:25.489023  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top2 className:harvester, reaper confidence:9.57031 classIndex:595
I0910 08:52:25.489029  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top3 className:doormat, welcome mat confidence:9.22656 classIndex:539
I0910 08:52:25.489035  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top4 className:sandbar, sand bar confidence:9.17969 classIndex:977
I0910 08:52:25.489040  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top5 className:radiator confidence:9.01562 classIndex:753
I0910 08:52:25.497282  3967 EfficientNetB5Classify.cpp:130] images crop_x1: 16, crop_x: 489, crop_y1: 16, crop_y: 489
I0910 08:52:25.538460  3967 Resnet50PostProcess.cpp:73] Start to Process Resnet50PostProcess.
I0910 08:52:25.539130  3967 Resnet50PostProcess.cpp:120] End to Process Resnet50PostProcess.
I0910 08:52:25.539146  3967 EfficientNetB5Classify.cpp:241] image path: /home/data/mindx/dataset/imagenet_val/ILSVRC2012_val_00014552.JPEG
I0910 08:52:25.539156  3967 EfficientNetB5Classify.cpp:243] file path for saving result: result/ILSVRC2012_val_00014552_1.txt
I0910 08:52:25.539247  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top1 className:doormat, welcome mat confidence:12.4922 classIndex:539
I0910 08:52:25.539270  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top2 className:chainlink fence confidence:11.8984 classIndex:489
I0910 08:52:25.539283  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top3 className:tile roof confidence:11.4453 classIndex:858
I0910 08:52:25.539289  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top4 className:window screen confidence:10.9688 classIndex:904
I0910 08:52:25.539304  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top5 className:pay-phone, pay-station confidence:9.88281 classIndex:707
I0910 08:52:25.541152  3967 EfficientNetB5Classify.cpp:130] images crop_x1: 16, crop_x: 489, crop_y1: 16, crop_y: 489
I0910 08:52:25.582245  3967 Resnet50PostProcess.cpp:73] Start to Process Resnet50PostProcess.
I0910 08:52:25.582909  3967 Resnet50PostProcess.cpp:120] End to Process Resnet50PostProcess.
I0910 08:52:25.582924  3967 EfficientNetB5Classify.cpp:241] image path: /home/data/mindx/dataset/imagenet_val/ILSVRC2012_val_00006604.JPEG
I0910 08:52:25.582933  3967 EfficientNetB5Classify.cpp:243] file path for saving result: result/ILSVRC2012_val_00006604_1.txt
I0910 08:52:25.583032  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top1 className:window shade confidence:13.2344 classIndex:905
I0910 08:52:25.583051  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top2 className:velvet confidence:11.3828 classIndex:885
I0910 08:52:25.583063  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top3 className:otter confidence:10.2734 classIndex:360
I0910 08:52:25.583071  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top4 className:doormat, welcome mat confidence:10.1016 classIndex:539
I0910 08:52:25.583081  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top5 className:window screen confidence:9.63281 classIndex:904
I0910 08:52:25.586199  3967 EfficientNetB5Classify.cpp:130] images crop_x1: 16, crop_x: 489, crop_y1: 16, crop_y: 489
I0910 08:52:25.627348  3967 Resnet50PostProcess.cpp:73] Start to Process Resnet50PostProcess.
I0910 08:52:25.628017  3967 Resnet50PostProcess.cpp:120] End to Process Resnet50PostProcess.
I0910 08:52:25.628031  3967 EfficientNetB5Classify.cpp:241] image path: /home/data/mindx/dataset/imagenet_val/ILSVRC2012_val_00016859.JPEG
I0910 08:52:25.628046  3967 EfficientNetB5Classify.cpp:243] file path for saving result: result/ILSVRC2012_val_00016859_1.txt
I0910 08:52:25.628121  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top1 className:park bench confidence:13.0469 classIndex:703
I0910 08:52:25.628139  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top2 className:velvet confidence:10.3984 classIndex:885
I0910 08:52:25.628146  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top3 className:window shade confidence:9.73438 classIndex:905
I0910 08:52:25.628154  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top4 className:paddle, boat paddle confidence:9.49219 classIndex:693
I0910 08:52:25.628162  3967 EfficientNetB5Classify.cpp:259] batchIndex:0 top5 className:swing confidence:9 classIndex:843
I0910 08:52:25.634276  3967 EfficientNetB5Classify.cpp:130] images crop_x1: 16, crop_x: 489, crop_y1: 16, crop_y: 489
I0910 08:52:25.675459  3967 Resnet50PostProcess.cpp:73] Start to Process Resnet50PostProcess.
I0910 08:52:25.676152  3967 Resnet50PostProcess.cpp:120] End to Process Resnet50PostProcess.
```


   ## MindX SDK推理

在容器内用SDK进行推理

首先进入/home/HwHiAiUser/EfficientNet-B5_ID1621_for_PyTorch/infer/ 路径下（文件所在路径需根据实际情况进行修改）。

   1. 编译后处理代码。

      由于该模型在后处理部分使用的是resnet50模型，因此在使用时无需编译，只需调用libresnet50postprocess.so文件即可。

   2. 修改配置文件。

      ```
      cd data
      vim efficientnetb5.pipeline
      ```

      打开efficientnetb5.pipeline文件，根据实际情况修改efficientnetb5.pipeline文件中图片规格、模型路径、配置文件路径和标签路径。

      ```
        "mxpi_imageresize0": {
                  "props": {
                      "handleMethod": "opencv",
                      "resizeType": "Resizer_Stretch",
                      "resizeHeight": "522",
                      "resizeWidth": "522"
                  },
                  "factory": "mxpi_imageresize",
                  "next": "mxpi_opencvcentercrop0"
              },
              "mxpi_opencvcentercrop0": {
                  "props": {
                      "dataSource": "mxpi_imageresize0",
                      "cropHeight": "488",
                      "cropWidth": "488"
                  },
                  "factory": "mxpi_opencvcentercrop",
                  "next": "mxpi_tensorinfer0"
              },
      "mxpi_tensorinfer0": {
                  "props": {
                      "dataSource": "mxpi_opencvcentercrop0",
                      "modelPath": "../data/efficientnetb5.om",
                      "waitingTime": "2000",
                      "outputDeviceId": "-1"
                  },
                  "factory": "mxpi_tensorinfer",
                  "next": "mxpi_classpostprocessor0"
              },
              "mxpi_classpostprocessor0": {
                  "props": {
                      "dataSource": "mxpi_tensorinfer0",
                      "postProcessConfigPath":"../data/efficientnetb5_sdk_infer.cfg",
                      "labelPath": "../data/imagenet1000_clsidx_to_labels.names",
                      "postProcessLibPath": "/usr/local/sdk_home/mxManufacture/lib/
                      modelpostprocessors/libresnet50postprocess.so"
                  },
                  "factory": "mxpi_classpostprocessor",
                  "next": "mxpi_dataserialize0"
              },
      ```

      打开main.py文件，根据实际情况修改main.py文件中打开pipeline文件的名称以及stream_name等设置。

      ```
      cd sdk
      vim main.py
      ```

      ```
       with open("efficientnetb5.pipeline", 'rb') as f:
              pipeline_str = f.read()
          ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
      
      
          if ret != 0:
              print("Failed to create Stream, ret=%s" % str(ret))
              exit()
      
          # Construct the input of the stream
          data_input = MxDataInput()
      
          dir_name = sys.argv[1]
          res_dir_name = sys.argv[2]
          file_list = os.listdir(dir_name)
          if not os.path.exists(res_dir_name):
              os.makedirs(res_dir_name)
      
          for file_name in file_list:
              file_path = os.path.join(dir_name, file_name)
              if file_name.endswith(".JPEG") or file_name.endswith(".jpeg"):
                  portion = os.path.splitext(file_name)
                  with open(file_path, 'rb') as f:
                      data_input.data = f.read()
              else:
                  continue
      
              empty_data = []
      
              stream_name = b'im_efficientnetb5'
      
      ```

      

   3. 运行推理服务。

      1.执行推理。

      ```
      cd sdk
      bash run.sh /home/data/mindx/dataset/imagenet_val/   ./result
      ```

      其中，/home/data/mindx/dataset/imagenet_val/为实际环境中使用的推理图片的路径；./result存放执行该命令得到的推理结果。

      2.查看推理结果。

      ```
      I0910 10:13:50.704779  4062 MxStreamManager.cpp:383] Creates stream(im_efficientnetb5) successfully.
      I0910 10:13:50.704820  4062 MxStreamManager.cpp:433] Creates streams successfully.
      W0910 10:13:50.802417  4096 MxsmStream.cpp:125] save result of unique id:0.
      sdk run time: 80391
      process img: ILSVRC2012_val_00047110.JPEG, infer result: {"MxpiClass":[{"classId":221,"className":"Irish water spaniel","confidence":19.9375},{"classId":219,"className":"cocker spaniel, English cocker spaniel, cocker","confidence":15.2109375},{"classId":266,"className":"miniature poodle","confidence":13},{"classId":267,"className":"standard poodle","confidence":12.109375},{"classId":184,"className":"Irish terrier","confidence":9.90625}]}
      W0910 10:13:50.880995  4096 MxsmStream.cpp:125] save result of unique id:1.
      sdk run time: 60284
      process img: ILSVRC2012_val_00014552.JPEG, infer result: {"MxpiClass":[{"classId":505,"className":"coffeepot","confidence":19.078125},{"classId":550,"className":"espresso maker","confidence":15.671875},{"classId":899,"className":"water jug","confidence":15.53125},{"classId":503,"className":"cocktail shaker","confidence":14.8984375},{"classId":967,"className":"espresso","confidence":13.3046875}]}
      W0910 10:13:50.931238  4096 MxsmStream.cpp:125] save result of unique id:2.
      sdk run time: 60318
      process img: ILSVRC2012_val_00006604.JPEG, infer result: {"MxpiClass":[{"classId":276,"className":"hyena, hyaena","confidence":22.40625},{"classId":275,"className":"African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus","confidence":17.453125},{"classId":341,"className":"hog, pig, grunter, squealer, Sus scrofa","confidence":12.1640625},{"classId":342,"className":"wild boar, boar, Sus scrofa","confidence":11.59375},{"classId":287,"className":"lynx, catamount","confidence":11.5234375}]}
      W0910 10:13:50.991847  4096 MxsmStream.cpp:125] save result of unique id:3.
      sdk run time: 60304
      process img: ILSVRC2012_val_00016859.JPEG, infer result: {"MxpiClass":[{"classId":2,"className":"great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias","confidence":20.75},{"classId":3,"className":"tiger shark, Galeocerdo cuvieri","confidence":19.390625},{"classId":5,"className":"electric ray, crampfish, numbfish, torpedo","confidence":13.078125},{"classId":394,"className":"sturgeon","confidence":12.9921875},{"classId":4,"className":"hammerhead, hammerhead shark","confidence":11.8828125}]}
      W0910 10:13:51.060590  4096 MxsmStream.cpp:125] save result of unique id:4.
      sdk run time: 60329
      process img: ILSVRC2012_val_00020009.JPEG, infer result: {"MxpiClass":[{"classId":336,"className":"marmot","confidence":19.609375},{"classId":371,"className":"patas, hussar monkey, Erythrocebus patas","confidence":15.1171875},{"classId":972,"className":"cliff, drop, drop-off","confidence":13.65625},{"classId":356,"className":"weasel","confidence":13.46875},{"classId":370,"className":"guenon, guenon monkey","confidence":11.6484375}]}
      W0910 10:13:51.122462  4096 MxsmStream.cpp:125] save result of unique id:5.
      sdk run time: 60333
      process img: ILSVRC2012_val_00025515.JPEG, infer result: {"MxpiClass":[{"classId":917,"className":"comic book","confidence":21.84375},{"classId":921,"className":"book jacket, dust cover, dust jacket, dust wrapper","confidence":20.484375},{"classId":454,"className":"bookshop, bookstore, bookstall","confidence":17.828125},{"classId":624,"className":"library","confidence":14.0859375},{"classId":446,"className":"binder, ring-binder","confidence":14.03125}]}
      W0910 10:13:51.176019  4096 MxsmStream.cpp:125] save result of unique id:6.
      sdk run time: 60365
      process img: ILSVRC2012_val_00046794.JPEG, infer result: {"MxpiClass":[{"classId":822,"className":"steel drum","confidence":13.28125},{"classId":412,"className":"ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin","confidence":12.609375},{"classId":504,"className":"coffee mug","confidence":12.1875},{"classId":427,"className":"barrel, cask","confidence":12.0625},{"classId":541,"className":"drum, membranophone, tympan","confidence":11.0546875}]}
      W0910 10:13:51.247227  4096 MxsmStream.cpp:125] save result of unique id:7.
      sdk run time: 60354
      
      ```

   4. 执行精度和性能测试。

      1.执行命令查看推理精度。

      ```
      cd util
      python3 classification_task_metric.py ../sdk/result/ /home/data/mindx/dataset/val_lable.txt ./ result.json
      ```

      参数说明：

      - [ ] 第一个参数（../sdk/result/）：推理结果保存路径。

      - [ ] 第二个参数（/home/data/mindx/dataset/val_lable.txt）：验证集标签文件。

      - [ ] 第三个参数（./）：精度结果保存目录。

      - [ ] 第四个参数（result.json）：结果文件。

        执行查看推理精度的命令涉及到的参数需根据实际环境进行修改。

      查看推理精度结果。

      ```
      vim result.json
      ```

​      2.性能统计。

​        a.打开性能统计开关。将“enable_ps”参数设置为true，“ps_interval_time”参数设置为6。		    

```
vim  /home/HwHiAiUser/mxVision/config/sdk.conf
```

```
# MindX SDK configuration file

# whether to enable performance statistics, default is false [dynamic config]
enable_ps=true
...
ps_interval_time=6
...
```

​		b.执行run.sh脚本。

```
bash run.sh 
```

​		c.在日志目录“/home/HwHiAiUser/mxVision/logs/”查看性能统计结果。

```
performance—statistics.log.e2e.xxx
performance—statistics.log.plugin.xxx
performance—statistics.log.tpr.xxx
```

​		其中，e2e日志统计端到端时间，plugin日志统计单插件时间。


# 在ModelArts上应用

## 上传自定义镜像

 登录 [SWR控制台](https://console.huaweicloud.com/swr/?agencyId=5b5810ebce86453a8f77ded5695374cd®ion=cn-north-4&locale=zh-cn&region=cn-north-4#/app/dashboard)，上传PyTorch训练镜像。具体请参见 [容器引擎客户端上传镜像](https://support.huaweicloud.com/qs-swr/)章节。 

![note_3.0-zh-cn](./images/note_3.0-zh-cn.png)  

SWR的区域需要与ModelArts所在的区域一致。例如：当前ModelArts在华北-北京四区域。SWR所在区域，请选择华北-北京四。 

## 创建OBS桶

1.创建桶。

​	登录[OBS管理控制台](https://storage.huaweicloud.com/obs)，创建OBS桶。具体请参见[创建桶](https://support.huaweicloud.com/usermanual-obs/obs_03_0306.html)章节。例如，创建名称为“pytorch-dataset”的OBS桶。

  ![note_3.0-zh-cn.png](./images/note_3.0-zh-cn.png)

创建桶的区域需要与ModelArts所在的区域一致。例如：当前ModelArts在华北-北京四区域，在对象存储服务创建桶时，请选择华北-北京四。 

2.创建文件夹存放数据。

创建用于存放数据的文件夹，具体请参见[新建文件夹](https://support.huaweicloud.com/usermanual-obs/obs_03_0316.html)章节。例如，在已创建的OBS桶“efficientnet-b5”中创建logs、outputs目录。

![img](./images/1837544.png)

目录结构说明：

- code：存放训练脚本目录
- dataset：存放训练数据集目录
- logs：存放训练日志目录
- model：存放om模型文件目录
- outputs：训练生成ckpt和pb模型目录

将EfficientNet-B5模型的代码文件夹直接上传至“code”目录，数据集imagenet2012传至“dataset”目录。

此步骤需要修改的文件：

（1）将modelarts文件夹下的train_net.py放在如图所示的EfficientNet-B5_for_PyTorch文件夹下面；

（2）将modelarts文件夹下的loader.py文件放在如图所示的EfficientNet-B5_for_PyTorch\pycls\datasets文件夹下面；

（3）将modelarts文件夹下的trainer.py放在如图所示的EfficientNet-B5_for_PyTorch\pycls\core文件夹下面；

（4）将modelarts文件夹下的config.py放在如图所示的EfficientNet-B5_for_PyTorch\pycls\core文件夹下面；

## 创建训练作业

1. 使用华为云帐号登录[ModelArts管理控制台](https://console.huaweicloud.com/modelarts)，在左侧导航栏中选择“训练管理 > 训练作业”，默认进入“训练作业”列表。

2. 单击“创建训练作业”，进入“创建训练作业”页面，在该页面填写训练作业相关参数。

3. 在创建训练作业页面，填写训练作业相关参数，然后单击“下一步”。

   本步骤只提供训练任务部分参数配置说明，其他参数配置详情请参见《[ModelArts AI 工程师用户指南](https://support.huaweicloud.com/modelarts/index.html)》中“使用自定义镜像训练模型”章节。

   a. 填写基本信息。

   ​	基本信息包含“名称”和“描述”。

   b. 填写作业参数。

   ![img](./images/2007448.png)

<table cellpadding="4" cellspacing="0" summary="" frame="border" border="1" rules="all">
      <caption>
       <b>表1 </b>部分作业参数说明
      </caption>
      <thead align="left">
       <tr>
        <th align="left" class="cellrowborder" valign="top" width="22.59225922592259%" id="mcps1.3.1.3.2.2.2.2.2.4.1.1"><p>参数名称</p> </th> 
        <th align="left" class="cellrowborder" valign="top" width="23.332333233323332%" id="mcps1.3.1.3.2.2.2.2.2.4.1.2"><p>子参数</p> </th> 
        <th align="left" class="cellrowborder" valign="top" width="54.07540754075407%" id="mcps1.3.1.3.2.2.2.2.2.4.1.3"><p>说明</p> </th> 
       </tr> 
      </thead> 
      <tbody>
       <tr>
        <td class="cellrowborder" valign="top" width="22.59225922592259%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.1 "><p>一键式参数配置</p> </td> 
        <td class="cellrowborder" valign="top" width="23.332333233323332%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.2 "><p>-</p> </td> 
        <td class="cellrowborder" valign="top" width="54.07540754075407%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.3 "><p>如果您在ModelArts已保存作业参数，您可以根据界面提示，选择已有的作业参数，快速完成训练作业的参数配置。</p> </td> 
       </tr> 
       <tr>
        <td class="cellrowborder" valign="top" width="22.59225922592259%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.1 "><p>算法来源</p> </td> 
        <td class="cellrowborder" valign="top" width="23.332333233323332%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.2 "><p>自定义</p> </td> 
        <td class="cellrowborder" valign="top" width="54.07540754075407%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.3 ">
         <ul>
          <li>镜像地址：镜像上传到SWR后生成的地址，例如：swr.cn-north-4.myhuaweicloud.com/<strong><em>user-job-dir</em></strong>/pytorch-ascend910-cp37-euleros2.8-aarch64:8.0。</li>
          <li>代码目录：训练代码文件存储的OBS路径，例如：/efficientnet-b5/EfficientNet-B5_for_PyTorch/。</li>
          <li>运行命令：镜像启动后的运行命令，例如：/bin/bash run_train.sh  'obs://efficientnet-b5/EfficientNet-B5_for_PyTorch/'  'EfficientNet-B5_for_PyTorch/train_net_ma.py'    '/tmp/log/training.log'   --cfg  '/home/work/user-job-dir/EfficientNet-B5_for_PyTorch/configs/dds_baselines/effnet/EN-B5_dds_1npu_full.yaml' --device_id 0  --npu 0  --data_url 'obs://efficientnet-b5/dataset/' --train_url 'obs://efficientnet-b5/outputs' OPTIM.MAX_EPOCH 3  TEST.DATASET imagenet2012 TRAIN.DATASET imagenet2012 TRAIN.BATCH_SIZE 12  </li>
         </ul> </td> 
       </tr> 
       <tr>
        <td class="cellrowborder" rowspan="2" valign="top" width="22.59225922592259%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.1 "><p>数据来源</p> </td> 
        <td class="cellrowborder" valign="top" width="23.332333233323332%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.2 "><p>数据集</p> </td> 
        <td class="cellrowborder" valign="top" width="54.07540754075407%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.3 "><p>从ModelArts数据管理中选择可用的数据集及其版本。</p> 
         <ul>
          <li>选择数据集：从右侧下拉框中选择ModelArts系统中已有的数据集。当ModelArts无可用数据集时，此下拉框为空。</li>
          <li>选择版本：根据“选择数据集”指定的数据集选择其版本。</li>
         </ul> </td> 
       </tr> 
       <tr>
        <td class="cellrowborder" valign="top" headers="mcps1.3.1.3.2.2.2.2.2.4.1.1 "><p>数据存储位置</p> </td> 
        <td class="cellrowborder" valign="top" headers="mcps1.3.1.3.2.2.2.2.2.4.1.2 "><p>从OBS桶中选择训练数据。在“数据存储位置”右侧，单击“选择”，从弹出的对话框中，选择数据存储的OBS桶及其文件夹。</p> </td> 
       </tr> 
       <tr>
        <td class="cellrowborder" valign="top" width="22.59225922592259%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.1 "><p>训练输出位置</p> </td> 
        <td class="cellrowborder" valign="top" width="23.332333233323332%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.2 "><p>-</p> </td> 
        <td class="cellrowborder" valign="top" width="54.07540754075407%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.3 "><p>选择训练结果的存储位置。</p> 
         <div class="note">
          <span class="notetitle"> 说明： </span>
          <div class="notebody">
           <p>为避免出现错误，建议选择一个空目录作“训练输出位置”。请勿将数据集存储的目录作为训练输出位置。</p> 
          </div>
         </div> </td> 
       </tr> 
       <tr>
        <td class="cellrowborder" valign="top" width="22.59225922592259%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.1 "><p>环境变量</p> </td> 
        <td class="cellrowborder" valign="top" width="23.332333233323332%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.2 "><p>-</p> </td> 
        <td class="cellrowborder" valign="top" width="54.07540754075407%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.3 "><p>请根据您的镜像文件，添加环境变量，此参数为可选。单击“增加环境变量”可增加多个变量参数。</p> </td> 
       </tr> 
       <tr>
        <td class="cellrowborder" valign="top" width="22.59225922592259%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.1 "><p>作业日志路径</p> </td> 
        <td class="cellrowborder" valign="top" width="23.332333233323332%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.2 "><p>-</p> </td> 
        <td class="cellrowborder" valign="top" width="54.07540754075407%" headers="mcps1.3.1.3.2.2.2.2.2.4.1.3 "><p>选择作业运行中产生的日志文件存储路径。</p> </td> 
       </tr> 
      </tbody>  
  </table>




​			c.选择用于训练作业的资源。

  			 此次项目选择资源类型为“Ascend”。

![img](./images/2008416.png)

d. 勾选“保存作业”。

e. 完成参数填写后，单击“下一步”。

4.在“规格确认”页面，确认填写信息无误后，单击“提交”，完成训练作业的创建。

训练作业一般需要运行一段时间，根据您选择的数据量和资源不同，训练时间将耗时几分钟到几十分钟不等。

## 查看训练任务日志

1. 在ModelArts管理控制台，选择“训练管理 > 训练作业”，进入训练作业列表页面。
2. 在训练作业列表中，单击作业名称，查看该作业的详情。
3. 选择“日志”页签，可查看该作业日志信息

![20210910181753](./images/20210910181753.png)

## 冻结模型

描述要点：在ModelArts上完成训练后export出对应onnx模型。

1. 训练生成pyth。

2. 将pyth导出成onnx。

## 迁移学习

1.准备数据集。

请参见“训练 > 迁移学习指导”，准备迁移学习训练所需的数据集，并上传到OBS的“obs://efficientnet-b5/dataset/”目录下。

![](images/0f7ce5c1fcea.png)

2.准备预训练模型model.pyth，由于前一项任务中已生成模型训练文件，因此将该模型文件作为迁移学习中的预训练模型。

![image-20210910182307083](./images/d16ffd6218e0.png)

3.创建训练作业，进行迁移学习。

请参考“创建训练作业”章节。

4.创建训练作业进行迁移学习。修改启动命令，增加以下参数：

- TRAIN.WEIGHTS  'obs://efficientnet-b5/outputs/model.pyth'

  在dataset数据集里面的train和val文件夹下分别增加两个数据集文件夹，即代码中的nun_classes增加了两类

![img](./images/de512997f83e.png)



5.在指定的训练输出位置查看生成的模型。

![img](./images/6238b8f1ee39.png)



