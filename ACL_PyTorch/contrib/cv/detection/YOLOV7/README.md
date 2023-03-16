
# YOLOV7模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

YOLO 算法作 为 one-stage 目标检测算法最典型的代表，其基于深度神经网络进行对象的识别和定位，运行速度很快，可以用于实时系统。YOLOV7 是目前 YOLO 系列最先进的算法，在准确率和速度上超越了以往的 YOLO 系列。

**YOLOV7 主要的贡献在于**：

1. 模型重参数化
   YOLOV7 将模型重参数化引入到网络架构中，重参数化这一思想最早出现于 REPVGG 中。
2. 标签分配策略
   YOLOV7 的标签分配策略采用的是 YOLOV5 的跨网格搜索，以及 YOLOX 的匹配策略。
3. ELAN 高效网络架构
   YOLOV7 中提出的一个新的网络架构，以高效为主。
4. 带辅助头的训练
   YOLOV7 提出了辅助头的一个训练方法，主要目的是通过增加训练成本，提升精度，同时不影响推理的时间，因为辅助头只会出现在训练过程中。

`参考论文：`[YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696v1)

​    

- 参考实现：

```
    url=https://github.com/WongKinYiu/yolov7.git
    branch=master
    commit_id=1cb8aa5
    model_name=yolov7
```




通过 Git 获取对应 commit_id 的代码方法如下：


```
    git clone {repository_url}        # 克隆仓库的代码
    cd {repository_name}              # 切换到模型的代码仓目录
    git checkout {branch/tag}         # 切换到对应分支
    git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
    cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- | -------- | --------------------------- | ------------ |
  | images   | RGB_FP32 | batchsize x 3 x 1280 x 1280 | NCHW         |

- 输出数据

  | 输出数据 | 大小          | 数据类型 | 数据排布格式 |
  | -------- | -------------  | -------- | ------------ |
  | output | 1 x 102000 x 85 | FLOAT32      | NCHW           |

# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.6.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以 CANN 版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   
    ```
    git clone https://github.com/WongKinYiu/yolov7.git
    cd yolov7
    git reset --hard 1cb8aa5
    ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```
   
## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。
   本模型支持 [coco2017](http://images.cocodataset.org/zips/val2017.zip) 验证集。
   用户需自行获取数据集，将 instances_val2017.json 文件和 val2017 文件夹解压并上传数据集到源码包路径下。
   coco2017 验证集所需文件目录参考（只列出该模型需要的目录）。
   
   数据集目录结构如下:

    ```
       |-- coco2017                // 验证数据集
           |-- instances_val2017.json    //验证集标注信息  
           |-- val2017             // 验证集文件夹
    ```

2. 数据预处理。

    在代码主目录将原始数据集转换为模型输入的数据。
    
    ```
    python yolov7_preprocess.py --image_src_path $img_path --bin_file_path $result_path
    ```
    
    详见下表
   
    | 参数        | 说明                                          |
    | ----------- | --------------------------------------------- |
    | img_path    | 数据集路径(./coco2017/val2017/)               |
    | result_path | 数据预处理得到的bin文件保存位置(./pre_result) |
    
    

## 模型推理<a name="section741711594517"></a>

1.  模型转换

    使用 PyTorch 将模型权重文件 .pth 转换为 .onnx 文件，再使用 ATC 工具将 .onnx 文件转为离线推理模型文件 .om 文件。

    a.  获取权重文件。
    
      [下载链接](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) 采用名称为 yolov7-e6.pth 的权重文件
    
    b. 导出 onnx 文件。

      将模型权重文件 .pt 转换为 .onnx 文件。 

      1).  将代码仓上传至服务器任意路径下如（如：/home/HwHiAiUser）。

      2).  进入代码仓目录并将 yolov7-e6.pt 移到当前目录下。
    
      3).  在代码主目录使用 export.py 导出 onnx 文件。
    
      4).  运行脚本：

        python export.py --weights ./yolov7-e6.pt --batch-size 1 --simplify --grid --topk-all 100 /
        --iou-thres 0.65 --conf-thres 0.001 --img-size 1280 1280 --max-wh 1280 
    
    | 参数         | 说明                               |
    | ------------ | ---------------------------------- |
    | --weights    | 权重模型文件                       |
    | --batch_size | batch大小(1、4、8、16、32、64)     |
    | --topk-all   | 每张图片上最多能检测出来物体的数量 |
    | --iou-thres  | 交并比阈值                         |
    | --iou-thres  | 置信度阈值                         |
    | --img-size   | 图片大小                           |
    
       5).   获得 yolov7-e6.onnx 文件。
    
    c. 使用 ATC 工具将 ONNX 模型转 OM 模型。
    
     1).  配置环境变量。                  
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
    
       说明：该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。
    
       详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
    
     2).  执行命令查看芯片名称型号（$\{chip\_name\}）。
    
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
    
     3).  执行 ATC 命令。
    
        atc --framework=5 \
            --model=yolov7-e6-bs1.onnx \
            --output=yolov7-e6-bs1 \
            --input_format=NCHW \
            --input_shape="images:1,3,1280,1280" \
            --soc_version=${chip_name}
    
    参数说明：
    
    -   --model：为 ONNX 模型文件。
    -   --framework：5代表 ONNX 模型。
    -   --output：输出的 OM 模型。
    -   --input\_format：输入数据的格式。
    -   --input\_shape：输入数据的 shape。
    -   --soc\_version：处理器型号。             
    
       运行成功后生成 yolov7-e6.om 文件。
    
2. 开始推理验证

   a. 使用 ais-infer 工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   b. 执行推理。

   运行 ais_infer 脚本。

       python3 -m ais_bench --model yolov7-e6_bs1.om  --input /home/hym/yolov7/YOLOV7/prep --output result --batchsize 1 --outfmt BIN
   
     参数说明：
   
   - --model：om 模型路径。
   - --input：数据预处理得到的 bin 文件。
   - --output：推理结果保存的目录。
   - --batchsize： batchsize 的大小 
   
      >**说明：**
      >执行 ais_infer 工具请选择与运行环境架构相同的命令。参数详情请参见[《ais_infer 推理工具使用文档》](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)
   
   c. 精度验证。
   
   在代码主目录进行精度计算
       python yolov7_postprocess.py  --result_path=result/2023_03_13-16_55_29 --img_path=./coco/instances_val2017.json
   
   参数说明：
   
   - --img_path：数据集路径。
   - --result_path：推理文件保存的位置。
   
   d.性能验证。
   
   可使用 ais_infer 推理工具的纯推理模式验证不同 batch_size 的 om 模型的性能，参考命令如下：
   
   ```
   python ${ais_infer_path}/ais_infer.py --model=${om_model_path} --loop=20 --batchsize=${batch_size}
   ```

   | 参数           | 说明               |
   | -------------- | ------------------ |
   | ais_infer_path | ais_infer文件路径  |
   | om_model_path  | 模型文件保存的位置 |
   | batchsize      | batchsize大小      |



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用 ACL 接口推理计算，性能参考下列数据。

精度：
| Precesion  |mAP50|
|---|---|
| 310p 精度 | 73.3% |
| 标杆精度 | 73.5% |

此处精度为 bs1 精度，bs1 和最优 bs 精度无差别

性能:

| 芯片型号    | Batch Size | 数据集         | 性能 |
| ----------- | ---------- | -------------- | ---- |
| Ascend310P3 | 1          | coco2017验证集 | 32   |
| Ascend310P3 | 4          | coco2017验证集 | 32   |
| Ascend310P3 | 8          | coco2017验证集 | 30   |
| Ascend310P3 | 16         | coco2017验证集 | 32   |
| Ascend310P3 | 32         | coco2017验证集 | 31   |
