# 交付件基本信息

**发布者（Publisher）**：Huawei

**应用领域（Application Domain）**：Image Classification

**版本（Version）**：1.1

**修改时间（Modified）**：2021.09.10

**大小（Size）**：39.76MB \(pth.tar\)/20 MB \(onnx\)/15MB \(om\)

**框架（Framework）**：PyTorch (1.5.0)

**模型格式（Model Format）**：pth.tar/onnx/om

**精度（Precision）**：O2(训练)、FP32（推理）

**处理器（Processor）**：昇腾910/昇腾310

**应用级别（Categories）**：Released

**描述（Description）：**基于Pytorch框架的GhostNet目标检测网络模型训练并保存模型，通过ATC工具转换，可在昇腾AI设备上运行

# 概述

## 简述

*描述要点（key）：该论文提供了一个全新的Ghost模块，旨在通过廉价操作生成更多的特征图。基于一组原始的特征图，作者应用一系列线性变换，以很小的代价生成许多能从原始特征发掘所需信息的“幻影”特征图（Ghost feature maps）。该Ghost模块即插即用，通过堆叠Ghost模块得出Ghost bottleneck，进而搭建轻量级神经网络——GhostNet。在ImageNet分类任务，GhostNet在相似计算量情况下Top-1正确率达75.7%，高于MobileNetV3的75.2%。

- 参考论文：[Han K, Wang Y, Tian Q, et al. Ghostnet: More features from cheap operations[C\]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 1580-1589.](https://arxiv.org/abs/1911.11907)

- 参考实现：[GitHub - huawei-noah/CV-Backbones: CV backbones including GhostNet, TinyNet and TNT, developed by Huawei Noah's Ark Lab.](https://github.com/huawei-noah/CV-Backbones)

通过Git获取对应commit_id的代码方法如下：

```
git clone {repository_url}     # 克隆仓库的代码
cd {repository_name}           # 切换到模型的代码仓目录
git checkout  {branch}         # 切换到对应分支
git reset --hard ｛commit_id｝  # 代码设置到对应的commit_id
cd ｛code_path｝                # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```

### 默认配置

- 训练数据集预处理：

  图像的输入尺寸为240*240

  随机裁剪图像尺寸

  随机水平翻转图像

  根据ImageNet数据集通用的平均值和标准偏差对输入图像进行归一化

- 测试数据集预处理

  图像的输入尺寸为240*240（将图像最小边缩放到274，同时保持宽高比，然后在中心裁剪图像）

  根据ImageNet数据集通用的平均值和标准偏差对输入图像进行归一化

- 训练超参

  -b 1024   
  --opt npufusedsgd 
  --epochs 40 

  --weight-decay 4e-5 
  --momentum 0.9 
  --sched cosine 
  -j 8 
  --warmup-lr 1e-6  
  --drop 0.2  
  --warmup-epochs 4 
  --amp  
  --lr 0.4  
  --clip-grad 2.0 
  --npu 0
  --num-classes 1000

### 准备工作

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
/home/HwHiAiUser/GhostNet_for_pytorch_{version}_code/     #此代码路径需根据实际路径进行替换
├── infer                # MindX高性能预训练模型新增  
│   └── README.md        # 离线推理文档
│   ├── convert          # 转换om模型命令，AIPP
│   │   ├──ghostnet_aipp.cfg
│   │   └──convert_om.sh
│   ├── data             # 包括模型文件、模型输入数据集、模型相关配置文件（如label、SDK的pipeline）
│   │   ├──ghostnet_sdk_infer.cfg
│   │   ├──ghostnet.pipeline
|	|	├──imagenet1000_clsidx_to_labels.names
│   ├── mxbase           # 基于mxbase推理
│   │   ├── src
│   │   │   ├── GhostNetClassify.cpp
│   │   │   ├── GhostNetClassify.h
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

   # 推理

-   **[准备推理数据](#准备推理数据.md)**  
-   **[模型转换](#模型转换.md)**  
-   **[mxBase推理](#mxBase推理.md)**  
-   **[MindX SDK推理](#MindX-SDK推理.md)**  

### 准备推理数据

准备模型转换和模型推理所需目录及数据。

准备推理数据

1. 下载源码包。（例如：单击“下载模型脚本”和“下载模型”，下载所需软件包。）

2. 将源码上传至推理服务器任意目录并解压（如：“/home/HwHiAiUser“）。

   ```
   # 在环境上执行
   unzip GhostNet_for_pytorch_{version}_code.zip
   cd {code_unzip_path}/GhostNet_for_pytorch_{version}_code/infer && dos2unix `find .`
   mkdir -p data/imagenet_val
   ```

   ![](images/note_3.0-zh-cn.png)

   - code_unzip_path：代码解压目录，以“/home/HwHiAiUser”为例。
   - version：为模型版本。

3. 准备数据。

   （准备用于推理的图片、数据集、模型文件、代码等，放在同一数据路径中，如：“/home/HwHiAiUser“。）

   示例：

   1.由于后续推理均在容器中进行，因此需要把用于推理的图片、数据集、模型文件、代码等均放在同一数据路径中，后续示例将以“/home/HwHiAiUser“为例。

```
├── infer                 # MindX高性能预训练模型新增  
│   ├── convert           # 转换om模型命令，AIPP
│   │   ├──ghostnet_aipp.cfg
│   │   └──convert_om.sh
│   ├── data              # 包括模型文件、模型输入数据集、模型相关配置文件（如label、SDK的pipeline）
│   │   ├──imagenet_val  	  # 推理所需的数据集
|   |   ├──ghostnet_sdk_infer.cfg
│   │   ├──ghostnet.pipeline
|	|	├──imagenet1000_clsidx_to_labels.names
│   ├── mxbase            # 基于mxbase推理
│   └── sdk               # 基于sdk.run包推理；如果是C++实现，存放路径一样
│   └── util             
│   │   └──classification_task_metric.py # 精度验证脚本
│   └──docker_start_infer.sh     # 启动容器脚本
```

​		2. ONNX模型可通过“模型训练”后转换生成或通过“下载模型”获取。

​		3.将Imagenet数据集val中的所有图片放到“infer/data/imagenet_val ”目录下。	

4.启动容器。

​	进入“infer“目录，执行以下命令，启动容器。

​	bash docker\_start\_infer.sh   docker\_image:tag   model\_dir

​	**表 2**  参数说明

| 参数         | 说明                                |
| ------------ | ----------------------------------- |
| docker_image | 基础镜像，可从Ascend Hub上下载      |
| tag          | 镜像tag，请根据实际配置，如：21.0.2 |
| data_dir     | 脚本路径                            |

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

   1. 准备ONNX模型文件。

      ONNX模型文件为在昇腾910服务器上导出的模型，导出ONNX模型的详细步骤请参考“模型训练”。将ONNX模型放到“GhostNet_for_pytorch_{version}_code/infer/data/”路径下。

      ```
      cd /home/HwHiAiUser/GhostNet_for_pytorch_{version}_code 
      ```

   2. 模型转换。

      进入“infer/convert/”目录进行模型转换，转换详细信息可查看转换脚本和对应的aipp配置文件，转换命令如下。

      ```
input_onnx_path=$1
      aipp_cfg_file=$2
      output_om_path=$3
      ...
      atc --input_format=NCHW \
          --framework=5 \                                # 5代表pytorch。
          --model="${input_onnx_path}" \                 # 待转换的onnx模型，模型可以通过训练生成或通过“下载模型”获得。
          --input_shape="image:1,3,240,240"  \           # 输入数据的shape。input取值根据实际使用场景确定。
          --output="${output_om_path}" \                 # 转换后输出的om模型。
          --insert_op_conf="${aipp_cfg_file}" \          # aipp配置文件。 
          --enable_small_channel=1 \                     # 是否使能small channel的优化。
          --log=error \                                  # 日志级别。
          --soc_version=Ascend310 \                      # 模型转换时指定芯片版本。
          --op_select_implmode=high_precision \          # 选择算子是高精度实现还是高性能实现。
      ```

    --output_type=FP32                             # om模型的数据类型
      ```
    
      转换命令如下。
      
      ```
bash convert_om.sh    input_onnx_path    aipp_cfg_file    output_om_path
      ```

      **表 1**  参数说明
      
      | 参数            | 说明                                              |
      | --------------- | ------------------------------------------------- |
      | input_onnx_path | onnx文件路径。                                    |
      | aipp_cfg_file   | aipp config文件路径。                             |
      | output_om_path  | 生成的OM文件名，转换脚本会在此基础上添加.om后缀。 |

   ## mxBase推理

#### 前提条件

已进入推理容器环境。

#### 操作步骤

1.配置环境变量。

```
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_VERSION=nnrt/latest
export ARCH_PATTERN=.
export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib/modelpostprocessors:$LD_LIBRARY_PATH
```

2.修改配置文件。

可根据实际情况修改，配置文件位于“mxbase/src/main.cpp”中，可修改参数如下。

```
...
const uint32_t CLASS_NUM = 1000;               # 实际的类别数量
const uint32_t ratio = 1000;
...
	InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = "./../data/imagenet1000_clsidx_to_labels.names";  # 类别标签文件根据实际情况修改
    initParam.topk = 1;
    initParam.softmax = false;
    initParam.checkTensor = true;
    initParam.modelPath = "./../data/ghostnet.om";   # 推理模型文件
...
```

3.编译工程。

目前mxBase推理仅实现了基于opencv方式推理。

```
cd /home/HwHiAiUser/GhostNet_for_pytorch_{version}_code/infer/mxbase
bash build.sh
```

4.运行推理服务。

​	1.指定图片目录进行推理，推理结果将打印在输出信息中。

```
./build/ghostnet   [image_path]
```

![](images/note_3.0-zh-cn.png)

[image_path]： 推理图片目录路径。可选用“../data/imagenet_val”

​	观察结果。

```
I0910 01:47:30.602972  3697 Resnet50PostProcess.cpp:73] Start to Process Resnet50PostProcess.
I0910 01:47:30.603637  3697 Resnet50PostProcess.cpp:120] End to Process Resnet50PostProcess.
I0910 01:47:30.603658  3697 GhostNetClassify.cpp:239] image path: /home/data/mindx/dataset/imagenet_val/ILSVRC2012_val_00042634.JPEG
I0910 01:47:30.603672  3697 GhostNetClassify.cpp:241] file path for saving result: result/ILSVRC2012_val_00042634_1.txt
I0910 01:47:30.603746  3697 GhostNetClassify.cpp:253] batchIndex:0 top1 className:croquet ball confidence:5.85621e+37 classIndex:522
I0910 01:47:30.610648  3697 GhostNetClassify.cpp:128] images crop_x1: 16, crop_x: 241, crop_y1: 16, crop_y: 241
I0910 01:47:30.616103  3697 Resnet50PostProcess.cpp:73] Start to Process Resnet50PostProcess.
I0910 01:47:30.616668  3697 Resnet50PostProcess.cpp:120] End to Process Resnet50PostProcess.

```

​	2.（可选）将“infer/util/”目录下的classification_task_metric.py 文件复制到“infer/mxbase”目录下	

```
python3 classification_task_metric.py result/ ./val_label.txt ./ ./result.json
```

​	参数说明：

- [ ] 第一个参数（result/）：推理结果保存路径。

- [ ] 第二个参数（./val_label.txt）：验证集标签文件。

- [ ] 第三个参数（./）：精度结果保存目录。

- [ ] 第四个参数（./result.json）：结果文件。

  查看推理精度结果。

```
cat result.json
```

   ## MindX SDK推理

在容器内用SDK进行推理

#### 前提条件

参见“准备推理数据”。

#### 操作步骤

1.修改配置文件。

​	a.可根据实际情况修改“infer/data/”目录下的pipeline文件。

```
{
    "im_ghostnet": {
        "stream_config": {
            "deviceId": "0"
        },
        "appsrc0": {
            "props": {
                "blocksize": "409600"
            },
            "factory": "appsrc",
            "next": "mxpi_imagedecoder0"
        },
        "mxpi_imagedecoder0": {
            "props": {
                "handleMethod": "opencv"
            },
            "factory": "mxpi_imagedecoder",
            "next": "mxpi_imageresize0"
        },
        "mxpi_imageresize0": {
            "props": {
                "handleMethod": "opencv",
                "resizeHeight": "274",
                "resizeWidth": "274",
                "resizeType": "Resizer_Stretch"
            },
            "factory": "mxpi_imageresize",
            "next": "mxpi_opencvcentercrop0"
        },
        "mxpi_opencvcentercrop0": {
            "props": {
                "dataSource": "mxpi_imageresize0",
                "cropHeight": "240",
                "cropWidth": "240"
            },
            "factory": "mxpi_opencvcentercrop",
            "next": "mxpi_tensorinfer0"
        },
        "mxpi_tensorinfer0": {
            "props": {
                "dataSource": "mxpi_opencvcentercrop0",
                "modelPath": "../data/ghostnet.om",   # 推理模型文件
                "waitingTime": "2000",
                "outputDeviceId": "-1"
            },
            "factory": "mxpi_tensorinfer",
            "next": "mxpi_classpostprocessor0"
        },
        "mxpi_classpostprocessor0": {
            "props": {
                "dataSource": "mxpi_tensorinfer0",
                "postProcessConfigPath":"../data/ghostnet_sdk_infer.cfg",   # 推理后处理相关配置
                "labelPath": "../data/imagenet1000_clsidx_to_labels.names",  # 推理数据集类别标签文件
                "postProcessLibPath": "/usr/local/sdk_home/mxManufacture/lib/modelpostprocessors/libresnet50postprocess.so"  # 推理后处理so文件
            },
            "factory": "mxpi_classpostprocessor",
            "next": "mxpi_dataserialize0"
        },
        "mxpi_dataserialize0": {
            "props": {
                "outputDataKeys": "mxpi_classpostprocessor0"
            },
            "factory": "mxpi_dataserialize",
            "next": "appsink0"
        },
        "appsink0": {
            "props": {
                "blocksize": "4096000"
            },
            "factory": "appsink"
        }
    }
}

```

​	从模型包中下载标签文件，并传入对应路径。标签文件参考路径为 “infer/data/imagenet1000_clsidx_to_labels.names”，如果	    	   	imagenet1000_clsidx_to_labels.names文件中第二行为unknown type，请删除该行。

```
# This is modified from https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
tench, Tinca tinca
goldfish, Carassius auratus
great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
tiger shark, Galeocerdo cuvieri
hammerhead, hammerhead shark
electric ray, crampfish, numbfish, torpedo
stingray
......
```

​	若使用迁移学习生成的模型进行推理，请修改imagenet1000_clsidx_to_labels.names文件的类别名称为迁移学习训练实际使用的类	  	别。示例如下。

```
person
bicycle
car
airplane
automobile
.....
```

​	b.可根据实际情况修改后处理配置文件。

​	若使用迁移学习生成的模型进行推理，请修改“infer/data/”目录下的配置文件ghostnet_aipp.cfg，CLASS_NUM为迁移学习训练任务数	据集的类别数量。

```
CLASS_NUM=1000 
SOFTMAX=true 
TOP_K=5
```

2.模型推理。

​	a.若要观测推理性能，需要打开性能统计开关。如下将“enable_ps”参数设置为true，“ps_interval_time”参数设置为6。

​		**vim** /usr/local/sdk_home/mxManufacture/config/sdk.conf

```
# MindX SDK configuration file

# whether to enable performance statistics, default is false [dynamic config]
enable_ps=true
...
ps_interval_time=6
...
```

​	b.进入“infer/sdk”目录，执行推理命令。

```
mkdir result  # 手动创建推理结果输出目录 
bash run.sh [image_path]   [infer_result]
```

![](images/note_3.0-zh-cn.png)

[image_path]： 推理图片目录路径。可选用“../data/imagenet_val”

 [infer_result] ：推理结果存放目录路径。可选用“./result”

​	查看推理结果。

```
I0910 02:53:15.640554  3755 MxStreamManager.cpp:383] Creates stream(im_ghostnet) successfully.
I0910 02:53:15.640640  3755 MxStreamManager.cpp:433] Creates streams successfully.
W0910 02:53:15.699101  3791 MxsmStream.cpp:125] save result of unique id:0.
sdk run time: 40236
process img: ILSVRC2012_val_00047110.JPEG, infer result: {"MxpiClass":[{"classId":822,"className":"steel drum","confidence":3.71521928e+18}]}
W0910 02:53:15.738744  3791 MxsmStream.cpp:125] save result of unique id:1.
sdk run time: 20127
```

​	c.查看推理性能和精度。

​		i.请确保性能开关已打开，在日志目录“/usr/local/sdk_home/mxManufacture/logs”查看性能统计结果。	

```
performance—statistics.log.e2e.xxx
performance—statistics.log.plugin.xxx
performance—statistics.log.tpr.xxx
```

​		其中e2e日志统计端到端时间，plugin日志统计单插件时间。

​		ii.将“infer/util/”目录下的classification_task_metric.py 文件复制到“infer/sdk”目录下。

​		执行命令计算推理精度。

```
python3 classification_task_metric.py result/ ./val_label.txt ./ ./result.json
```

​	参数说明：

- [ ] 第一个参数（result/）：推理结果保存路径。

- [ ] 第二个参数（./val_label.txt）：验证集标签文件。

- [ ] 第三个参数（./）：精度结果保存目录。

- [ ] 第四个参数（./result.json）：结果文件。

  查看推理精度结果。

```
cat result.json	
```

# 在ModelArts上应用

## 上传自定义镜像

1.从昇腾镜像仓库获取自定义镜像[ascend-pytorch-arm-modelarts](https://ascendhub.huawei.com/#/detail/ascend-pytorch-arm-modelarts)。

2.登录[SWR控制台](https://console.huaweicloud.com/swr/?agencyId=5b5810ebce86453a8f77ded5695374cd®ion=cn-north-4&locale=zh-cn&region=cn-north-4#/app/dashboard)，上传PyTorch训练镜像。具体请参见[容器引擎客户端上传镜像](https://support.huaweicloud.com/qs-swr/)章节。

![note_3.0-zh-cn](./images/note_3.0-zh-cn.png)  

SWR的区域需要与ModelArts所在的区域一致。例如：当前ModelArts在华北-北京四区域。SWR所在区域，请选择华北-北京四。 

## 创建OBS桶

1.创建桶。

登录[OBS管理控制台](https://storage.huaweicloud.com/obs)，创建OBS桶。具体请参见[创建桶](https://support.huaweicloud.com/usermanual-obs/obs_03_0306.html)章节。例如，创建名称为“ghostnet-pytorch”的OBS桶。

  ![note_3.0-zh-cn.png](./images/note_3.0-zh-cn.png)

创建桶的区域需要与ModelArts所在的区域一致。例如：当前ModelArts在华北-北京四区域，在对象存储服务创建桶时，请选择华北-北京四。 

2.创建文件夹存放数据。

创建用于存放数据的文件夹，具体请参见[新建文件夹](https://support.huaweicloud.com/usermanual-obs/obs_03_0316.html)章节。例如，在已创建的OBS桶“pytorch-dataset”中创建log、output目录。

![img](./images/1837544.png)

目录结构说明：

- code：存放训练脚本目录
- data：存放训练数据集目录
- logs：存放训练日志目录
- outputs：训练生成pth.tar和onnx模型目录

将GhostNet模型的代码文件夹直接上传至“code”目录，数据集imagenet传至“data”目录。

## 创建训练作业

1. 使用华为云帐号登录[ModelArts管理控制台](https://console.huaweicloud.com/modelarts)，在左侧导航栏中选择“训练管理 > 训练作业”，默认进入“训练作业”列表。

2. 在训练作业列表中，单击左上角“创建”，进入“创建训练作业”页面。

3. 在创建训练作业页面，填写训练作业相关参数，然后单击“下一步”。

   本步骤只提供训练任务部分参数配置说明，其他参数配置详情请参见《[ModelArts AI 工程师用户指南](https://support.huaweicloud.com/modelarts/index.html)》中“使用自定义镜像训练模型”章节。

   a. 填写基本信息。

   ​    设置训练作业名称。

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
          <li>镜像地址：镜像上传到SWR后生成的地址，例如：swr.cn-north-4.myhuaweicloud.com/hw94403450_00324175/torch-c76:5.0。</li>
          <li>代码目录：训练代码文件存储的OBS路径，例如：/pytorch-dataset/code/GhostNet_for_PyTorch/。</li>
          <li>运行命令：镜像启动后的运行命令，例如：/bin/bash run_train.sh 'obs://pytorch-dataset/code/GhostNet' 'GhostNet/train_start.py' '/tmp/log/training.log' --data_url='obs://pytorch-dataset/data/imagenet2012' --train_url='obs://pytorch-dataset/output/ghostnet/' --model GhostNet -b 512 --opt npufusedsgd --weight-decay 4e-5 --momentum 0.9 --sched cosine -j 8 --warmup-lr 1e-6 --drop 0.2 --warmup-epochs 4 --amp --lr 0.4 --clip-grad 2.0 --npu 0 --num-classes 1000 --epochs 1 /pytorch-dataset/data/imagenet2012 。</li>
          <div class="note">
          	<span class="notetitle"> 说明： </span>
          	<div class="notebody">
            <p>将modelarts文件夹下的train_start.py复制到与train_ghostnet_1p.py同级目录下。</p> 
          </div>
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




​		c.选择用于训练作业的资源。

  			 此次项目选择资源类型为“Ascend”。

![img](./images/2008416.png)

​		d. 勾选“保存作业”。

​		e. 完成参数填写后，单击“下一步”。

4.在“规格确认”页面，确认填写信息无误后，单击“提交”，完成训练作业的创建。

训练作业一般需要运行一段时间，根据您选择的数据量和资源不同，训练时间将耗时几分钟到几十分钟不等。

## 查看训练任务日志

1. 在ModelArts管理控制台，选择“训练管理 > 训练作业”，进入训练作业列表页面。
2. 在训练作业列表中，单击作业名称，查看该作业的详情。
3. 选择“日志”页签，可查看该作业日志信息

![img](./images/b214dfb3a.png)

## 迁移学习

1.准备数据集。

请参见“训练 > 迁移学习指导”，准备迁移学习训练所需的数据集，并上传到OBS的“pytorch-dataset/data”目录下。

![img](./images/d13c52a4.png)

2.准备预训练模型model_best.pth.tar，将预训练模型上传到outputs目录下。

![img](./images/9264ac50cc59.png)

3.准备训练代码。

请参考“创建训练作业”章节。

![img](./images/e350857037eb.png)

4.创建训练作业进行迁移学习。

​	   在原命令基础上，增加以下参数：

- --pretrained

- --pretrained_weight=" obs://ghostnet-pytorch/outputs/model_best.pth.tar"  

  修改以下命令参数：

- --num-classes 10

- /pytorch-dataset/dataset/cifar10/

- --data_url='obs://pytorch-dataset/dataset/cifar10/'

5.在指定的训练输出位置查看生成的模型。

![img](./images/e2df9146f00a.png)





