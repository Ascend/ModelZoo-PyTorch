# YOLOR模型-推理指导


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

yolor将统一网络的隐性知识(implicit knowledge)和显性知识(explicit knowledge)编码在一起，该网络可以生成一个统一的表征(representation)同时用于多个任务。可以利用模型学习的隐式知识来执行目标检测之外的广泛任务，且隐式知识有助于所有任务的性能提升。


- 参考实现：

  ```
  url=https://github.com/WongKinYiu/yolor
  branch=main 
  commit_id=b168a4dd0fe22068bb6f43724e22013705413afb
  model_name=yolor_p6
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
  | input    | RGB_FP32 | batchsize x 3 x 1344 x 1344 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | batchsize x 112455 x 85 | FLOAT32  | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                     | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.7.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

1. 获取源码：

  ```
    git clone https://github.com/WongKinYiu/yolor
    cd yolor
    git apply < ../yolor.patch
    cd ..                 # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   该模型使用coco数据集，该模型使用coco数据集，可根据https://github.com/WongKinYiu/yolor/blob/main/scripts/get_coco.sh 获取，
   **修改解压目录或者解压完成后移动coco置于此readme同级目录**。目录结构如下：
   ```
    ├── coco
    │    ├── images   
    │         ├── val2017   
    │    ├── labels
    │         ├── val2017
    ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行yolor_preprocess.py脚本，完成预处理。在本目录val2017_bin文件夹下生成bin文件。

   ```
   python3 yolor_preprocess.py --save_path ./val2017_bin --data ./coco.yaml --img_size 1344 --batch_size 1

   ```
   - 参数说明：
       -   --save_path：预处理后数据保存路径。
       -   --data：输入数据路径。
       -   --img_size：图像大小。
       -   --batch_size：batch大小。 
   
    获得val2017_bin下的bin文件。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       从源码包中获取权重文件：“yolor_p6.pt”。
       或者从此处下载https://drive.google.com/u/0/uc?id=1Tdn3yqpZ79X7R1Ql0zNlNScB1Dv9Fp76&export=download

   2. 导出onnx文件。

      1. 使用yolor_p6.pt导出onnx文件。

         运行yolor_pth2onnx.py脚本。

         ```
         python3 yolor_pth2onnx.py --cfg ./yolor_p6_swish.cfg --weights ./yolor_p6.pt --output_file ./yolor_bs1.onnx --batch_size 1 --img_size 1344
         ```
         - 参数说明：
        
           -   --cfg：配置文件。
           -   --weights：模型权重文件。
           -   --output_file：输出的onnx模型。
           -   --batch_size：batch大小。 
           -   --img_size：图像大小。
         
          获得yolor_bs1.onnx文件。

      2. 优化ONNX文件。

         ```
         python3 -m onnxsim --input-shape='1,3,1344,1344' yolor_bs1.onnx yolor_bs1_sim.onnx
         ```
         - 参数说明：
        
           -   --input-shape：输入数据的shape。
           -   yolor_bs1.onnx：输入模型。
           -   yolor_bs1_sim.onnx：输出优化后的模型。
          
         获得yolor_bs1_sim.onnx文件。

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
         使用atc将onnx模型转换为om模型文件，工具使用方法可以参考CANN V100R020C10 开发辅助工具指南 (推理) 01，需要指定输出节点以去除无用输出，可以使用netron开源可视化工具查看具体的输出节点名。

         这里只保留模型的output(type: float32[1,112455,85])一个输出，其前一个算子为Concat_1059：

         ```
         atc --model=yolor_bs1_sim.onnx --framework=5 --output=yolor_bs1 --input_format=NCHW --input_shape="image:1,3,1344,1344" --log=info --soc_version=Ascend${chip_name} --out_nodes="Concat_1059:0" --buffer_optimize=off_optimize
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --out\_nodes=输出节点。
           -   --buffer\_optimize=是否开启buffer优化。

           运行成功后生成<u>***yolor_bs1.om***</u>模型文件。



2. 开始推理验证。


a.  使用ais-infer工具进行推理。

 
   ais_infer工具获取及使用方式请点击查看《[ais_infer推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)》。


b.  执行推理。

```
 python3 ais_infer.py  --model /home/yolor/yolor_bs1.om --input /home/yolor/val2017_bin/ --output ./ --batchsize 1
```

  -  参数说明：
     -  --model：输入om文件路径。
     -  --input：输入bin文件的文件夹路径。
     -  --output：推理结果输出路径(如下精度验证输入)。
     -  --batchsize：默认为1。
             
c.  精度验证。

调用yolor_postprocess.py，可以获得Accuracy数据。修改yolor_postprocess.py第109行output_path为ais_infer推理的output路径。

```
 python3 yolor_postprocess.py --data ./coco.yaml --img 1344 --batch 1 --conf 0.001 --iou 0.65 --npu 0 --name yolor_p6_val --names ./yolor/data/coco.names
```

   -  参数说明：
        -  --data：输入数据路径。
         -  --img：图像大小。
         -  --batch：batch大小。
         -  --conf：置信度。
         -  --iou：交并比。
         -  --npu：npu。
         -  --name：保存路径。
         -  --names：数据集标签名。
    
执行完后会打印出精度:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.521
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.705
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.573
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.370
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.568
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.648
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.388
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.649
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.711
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.579
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.748
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.830
```
对比官网精度：
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.52510
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.70718
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.57520
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.37058
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.56878
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.66102
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.39181
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.65229
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.71441
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.57755
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.75337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.84013
```
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

T4执行推理。
```
trtexec --onnx=yolor_bs1.onnx --fp16 --threads --workspace=50000
```
 -  参数说明：
    -  --onnx：输入onnx模型。
     -  --fp16：fp16 精度。
     -  --threads： 启用多线程。
     -  --workspace：工作区大小。 
        
获得T4推理性能。

调用ACL接口推理计算，性能参考下列数据。

精度：

| Precision  | 310 |310P   |源码仓精度   |
| --------- | ---------------- |---------------- |---------------- |
|     AP      |      0.521            | 0.521 |    0.525 |


性能：

| Throughput |310   | 310P | T4 | 310P/310 | 310P/T4 |
| --------- | ---------------- | ---------- | ---------- | --------------- |--------------- |
|     bs1      |      32.333892           |      39.767756      |    32.60260        |        1.229909         | 1.219773  |
|     bs4      |       31.728015           |      40.488616      |   36.14153         |       1.276116          | 1.120280  |
|     bs8      |        31.810378          |     40.973069       |      34.11863      |         1.288041        | 1.200900  |
|     最优batch      |       32.333892            |      40.973069      |    36.14153         |      1.288041           | 1.219773  |

最优batch：310P的最优batch性能 >=1.2倍310最优batch性能 ，性能达标

batch_size超过8，由于模型过大无法推理。


