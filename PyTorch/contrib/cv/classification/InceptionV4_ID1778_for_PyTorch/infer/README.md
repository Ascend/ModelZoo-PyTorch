# 交付件基本信息

**发布者（Publisher）**：Huawei

**应用领域（Application Domain）**：Image Classification

**版本（Version）**：1.2

**修改时间（Modified）**：2021.09.10

**大小（Size）**：onnx/163MB \(om\)

**框架（Framework）**：PyTorch (1.5.0)

**模型格式（Model Format）**：pth.tar/onnx/om

**精度（Precision）**：O2(训练)、FP16（推理）

**处理器（Processor）**：昇腾910/昇腾310

**应用级别（Categories）**：Released

**描述（Description）：**基于Pytorch框架的InceptionV4图像分类网络模型训练并保存模型，通过ATC工具转换，可在昇腾AI设备上运行

**CANN 版本**:  CANN 5.0.2 B058

**Python 版本**: Python 3.7.5

**操作系统版本**:  Ubuntu 18.04

# 推理

### 操作步骤

1. 单击“下载模型脚本”和“下载模型”，下载所需软件包。

   

   ![img](https://r.huaweistatic.com/s/ascendstatic/lst/modelZooImg/public_sys-resources/note_3.0-zh-cn.png) 

   - 下载模型脚本：下载训练和推理的脚本文件。
   - 下载模型：下载模型文件。

   

2. 将源码上传至推理服务器任意目录并解压（如：“/home/test”）。

   

   ```
   # 在环境上执行
   unzip InceptionV4_ID1778_for_PyTorch.zip
   cd {code_unzip_path}/InceptionV4_ID1778_for_PyTorch/infer
   ```

   ![img](https://r.huaweistatic.com/s/ascendstatic/lst/modelZooImg/public_sys-resources/note_3.0-zh-cn.png) 

   - code_unzip_path：代码解压目录，以“/home/test”为例。
   - version：为模型版本。

   

   

3. 数据准备。

   

   由于后续推理均在容器中进行，因此需要把用于推理的图片、数据集、模型文件、代码等均映射到容器中

   ```
   /home/data
   ├── imagenet_val      # 推理图片用于精度测试的数据集
   │   ├── ILSVRC2012_val_00016652.JPEG   
   │   └── ILSVRC2012_val_00024989.JPEG
   ```

   

   

4. 文件格式转换

 ```
 find ./ -name "*.sh" | xargs dos2unix   
 ```

   

5. 启动容器
  

   进入到代码根目录，执行以下命令，启动容器。

   **bash scripts/docker_start_infer.sh** *infer_image* *data_path*

   | 参数          | 说明                          |
   | ------------- | ----------------------------- |
   | *infer_image* | 推理镜像名称，根据实际写入。  |
   | *data_path*   | 数据路径。如：“/home/test/”。 |

   启动容器时会将推理芯片和数据路径挂载到容器中。

   ```
   # 切换到代码跟目录下
   cd /home/path/to/InceptionV4_ID1778_for_PyTorch/
   bash scripts/docker_start_infer.sh infer:21.0.1 /home/path/to/path
   ```

## 模型转换

### 前提条件

已进入推理容器环境。具体操作请参见“准备容器环境”。

### 操作步骤

1. 准备onnx模型文件。

   

   onnx模型为在昇腾910服务器上导出的模型，导出onnx模型的详细步骤请参见“模型训练”。

   将模型文件保存在/path/to/InceptionV4_ID1778_for_PyTorch/infer/data 路径下

   

2. 执行以下命令，进行模型转换。

   

   模型转换时，可选择不同的AIPP配置文件进行转换。转换详细信息可查看转换脚本和对应的AIPP配置文件，转换命令如下。

   **bash convert/onnx2om.sh** model_path output_model_name

   | 参数              | 说明                                              |
   | ----------------- | ------------------------------------------------- |
   | model_path        | 转换脚本onnx文件路径。                            |
   | output_model_name | 生成的OM文件名，转换脚本会在此基础上添加.om后缀。 |

   转换示例如下所示。

   ```
   bash convert/onnx2om.sh ./data/InceptionV4_npu_1.onnx data/InceptionV4-pt
   ```

## mxBase推理

### 前提条件

已进入推理容器环境。具体操作请参见“准备容器环境”。

### 操作步骤

1. 配置环境变量。

   ```
   export ASCEND_HOME=/usr/local/Ascend
   export ASCEND_VERSION=nnrt/latest
   export ARCH_PATTERN=.
   export LD_LIBRARY_PATH=${MX_SDK_HOME}/lib/modelpostprocessors:${LD_LIBRARY_PATH}
   ```

   

2. 修改配置文件。

   

   可根据实际情况修改，配置文件位于“mxbase/src/main.cpp”中，可修改参数如下。

   若使用迁移学习生成的模型进行推理，请修改CLASS_NUM为迁移学习训练任务数据集的类别数量。如：修改CLASS_NUM=1001；om的模型路径；labelPath的路径

   ```
   ...
   namespace {
   
   const uint32_t CLASS_NUM = 1001;    // 推理类别总数
   }
   int main(int argc, char* argv[]){
       ....
       initParam.labelPath = "../data/imagenet1000_clsidx_to_labels.names";
       initParam.topk = 5;
       initParam.softmax = false;
       initParam.checkTensor = true;
       initParam.modelPath = "../data/inception-pt.om";
       ....
   }
   ...
   ```

   

3. 编译工程。

   ```
   cd /home/path/to/InceptionV4_ID1778_for_PyTorch/infer/mxbase
   bash build.sh
   ```

4. 运行推理服务

   

   **./build/inceptionv4    *image_path*

   | 参数       | 说明                                           |
   | ---------- | ---------------------------------------------- |
   | image_path | 推理图片所在文件的路径。如：“../data/images”。 |

   

5. 观察结果。

   分类结果会以*.txt的格式保存在result文件中。
   
   ![image-20210929155913474](./image/image-20210929155913474.png)
   
   

## MindX SDK 推理

### 前提条件

已进入推理容器环境。具体操作请参见“准备容器环境”。

### 操作步骤

1. 修改配置文件。可根据实际情况修改pipeline文件。

   ```
   vim ../data/inceptionv4_opencv.pipeline
   ```

   以inceptionv4_opencv.pipeline文件为例，作简要说明。

   ```
   {
       "im_inceptionv4": {
           "stream_config": {
               "deviceId": "0"    # 指定推理卡，要与挂载到容器中的卡一致
           },
           "appsrc0": {
               "props": {
                   "blocksize": "409600"
               },
               "factory": "appsrc",
               "next": "mxpi_imagedecoder0"   
           },
           ""mxpi_imagedecoder0"": {
               "props": {
                   "handleMethod": "opencv"
               },
               "factory": "mxpi_imagedecoder",
               "next": "mxpi_imageresize0"   
           },
           "mxpi_imageresize0": {
               "props": {
                   "handleMethod": "opencv",
                   "resizeHeight": "355",
                   "resizeWidth": "355",
                   "resizeType": "Resizer_Stretch"
               },
               "factory": "mxpi_imageresize",
               "next": "mxpi_opencvcentercrop0"
           },
          "mxpi_opencvcentercrop0": {
              "props": {
                   "dataSource": "mxpi_imageresize0",
                   "cropHeight": "299",
                   "cropWidth": "299"
               },
               "factory": "mxpi_opencvcentercrop",
               "next": "mxpi_tensorinfer0"
            },
   
           "mxpi_tensorinfer0": {
               "props": {
                   "dataSource": "mxpi_opencvcentercrop0",
                   "modelPath": "../data/inceptionV4_pt_cfg.om",  #模型存放路径
                   "waitingTime": "2000",
                   "outputDeviceId": "-1"
               },
               "factory": "mxpi_tensorinfer",
               "next": "mxpi_classpostprocessor0"
           },
           "mxpi_classpostprocessor0": {
               "props": {
                   "dataSource": "mxpi_tensorinfer0",
                   "postProcessConfigPath": "../data/inceptionv4_aipp.cfg", #后处理的配置文件
                   "labelPath": "../data/imagenet1000_clsidx_to_labels.names", #标签路径
                   "postProcessLibPath": "../../../../mxVision-2.0.2/lib/modelpostprocessors/libresnet50postprocess.so"   #后处理模块
               },
               "factory": "mxpi_classpostprocessor",
               "next": "mxpi_dataserialize0"
           },
   
           ....
       }
   }
   ```

   

2. 运行推理服务。

   ```
   cd infer/sdk
   bash run.sh /path/to/testImageDir /path/to/saveResultDir
   ```

   setp 1、查看推理结果。

   若设置推理结果路径为“infer/sdk/result”，示例如下所示。

   ```
   root@97a4c6ab6482:/home/path/to/InceptionV4_ID1778_for_PyTorch/infer/sdk/result# ll
   -rw-r--r-- 1 root   root          21 Aug 24 07:09 ILSVRC2012_val_00042663_1.txt
   -rw-r--r-- 1 root   root          21 Aug 24 07:09 ILSVRC2012_val_00042820_1.txt
   -rw-r--r-- 1 root   root          21 Aug 24 07:09 ILSVRC2012_val_00042855_1.txt
   -rw-r--r-- 1 root   root          21 Aug 24 07:09 ILSVRC2012_val_00043055_1.txt
   -rw-r--r-- 1 root   root          21 Aug 24 07:09 ILSVRC2012_val_00043439_1.txt
   -rw-r--r-- 1 root   root          21 Aug 24 07:09 ILSVRC2012_val_00044460_1.txt
   ```

3. 性能统计

   step 1、打开性能统计开关。将“enable_ps”参数设置为true，“ps_interval_time”参数设置为6

   **vi** */home/HwHiAiUser/mxManufacture/config/sdk.conf*

   ```
   # MindX SDK configuration file
   
   # whether to enable performance statistics, default is false [dynamic config]
   
   enable_ps=true
   ...
   ps_interval_time=6
   ...
   ```

   step 2、执行run.sh脚本。

   ```
   cd infer/sdk
   bash run.sh /path/to/testImageDir /path/to/saveResultDir
   ```

   step 3、在日志目录 “/home/HwHiAiUser/mxManufacture/logs/” 查看性能统计结果。

   ```
   performance—statistics.log.e2e.xxx
   performance—statistics.log.plugin.xxx
   performance—statistics.log.tpr.xxx
   ```

   其中e2e日志统计端到端时间，plugin日志统计单插件时间。

4.  执行精度测试。

 精度结果是在imageNet上进行的，使用classification_task_metric.py 进行测试。修改classification_task_metric.py

```
python3 classfication_task_metric.py result/ ./val_label.txt . ./result.json"
```

参数 1 ： prediction file path
参数 2 ： ground truth file
参数 3 ： result store path
参数 4 ： json file name 

查看精度结果。

```
cat result.json
```



![image-20210929111906059](./image/image-20210929111906059.png)
