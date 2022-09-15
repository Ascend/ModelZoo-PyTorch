
# YOLOV3模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

YOLOv3是一种端到端的one-stage目标检测模型。相比与YOLOv2，YOLOv3采用了一个新的backbnone——Darknet-53来进行特征提取工作，这个新网络比Darknet-19更加强大也比ResNet-101或者ResNet-152更加高效。同时，对于一张输入图片，YOLOv3可以在3个不同的尺度上预测物体框，每个尺度预测三种大小的边界框。通过这种多尺度联合预测的方式有效提升了小目标的检测精度。

`参考论文：Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement. arXiv 2018." arXiv preprint arXiv:1804.02767 (2018): 1-6.`

    

- 参考实现：

    
```
    url=https://github.com/ultralytics/yolov3.git
    branch=master
    commit_id=166a4d590f08b55cadfebc57a11a52ff2fc2b7d3
    model_name=yolov3
```



    
通过Git获取对应commit_id的代码方法如下：

    
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
  | -------- | -------- | ------------------------- | ------------ |
  | images   | RGB_FP32 | batchsize x 3 x 416 x 416 | NCHW         |

- 输出数据

  | 输出数据 | 大小          | 数据类型 | 数据排布格式 |
  | -------- | -------------  | -------- | ------------ |
  | Reshape_216  | 3x85x13x13 | FLOAT32      | NCHW           |
  | Reshape_203  | 3x85x26x26 | FLOAT32      | NCHW           |
  | Reshape_187  | 3x85x52x52 | FLOAT32      | NCHW           |

# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.6.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \          |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

    
    ```
        git clone https://github.com/ultralytics/yolov3.git
        cd yolov3
        git reset --hard 166a4d590f08b55cadfebc57a11a52ff2fc2b7d3
    ```

       
2. 安装依赖。

   ```
       pip3 install -r requirements.txt
   ```
   
## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。
   本模型支持coco2014验证集。
   用户需自行获取数据集，将instances_val2014.json文件和val2014文件夹解压并上传数据集到源码包路径下。
   coco2014验证集所需文件目录参考（只列出该模型需要的目录）。数据集下载链接(http://images.cocodataset.org/zips/val2014.zip)
    
   数据集目录结构如下:

    ```
       |-- coco2014                // 验证数据集
           |-- instances_val2014.json    //验证集标注信息  
           |-- val2014             // 验证集文件夹
    ```

   
2. 数据预处理。

    a.生成coco_2014.info数据集信息文件，使用parse_json.py脚本解析coco数据集中的json文件。
    创建ground-truth-split目录并执行parse_json.py脚本。

    ```
        mkdir ground-truth-split
        python3.7 parse_json.py
    ```
   
    执行成功后，在当前目录下生成coco2014.names和coco_2014.info文件以及标签文件夹ground-truth-split。
    
    说明：需要用户创建ground-truth-split文件夹。
    
    b.数据预处理将原始数据（.jpg）转化为模型输入的二进制文件（.bin）。
    执行preprocess_yolov3_pytorch.py脚本。

   ```
        python3.7 preprocess_yolov3_pytorch.py yolov3_bin
   ```
   
    参数说明：
        coco_2014.info数据集信息。
        yolov3_bin生成的二进制文件路径。

3. 生成数据集info文件。
    
    二进制输入info文件生成。
    使用get_coco_info.py脚本，输入已经得到的二进制文件，输出生成二进制数据集的info文件。
    
    运行get_coco_info.py脚本。
 
            python3.7 get_coco_info.py yolov3_bin 
       
    第一个参数为生成的数据集bin文件夹路径，第二个参数为数据集图片info文件。

## 模型推理<a name="section741711594517"></a>

1.  模型转换

    使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

    a.  获取权重文件。
     
     从源码包中获取训练后的权重文件yolov3.pt。
     
     源码包下载链接(https://www.hiascend.com/zh/software/modelzoo/models/detail/1/36ea401e0d844f549da2693c6289ad89)

    b. 导出onnx文件。

      将模型权重文件.pt转换为.onnx文件。 

      1).  将代码仓上传至服务器任意路径下如（如：/home/HwHiAiUser）。
      
      2).  进入代码仓目录并将yolov3.pt移到当前目录下。
      
      3).  修改models/export.py脚本，将转化的onnx算子版本设置为11。

            
            torch.onnx.export(model, img, f, verbose=True, 
                                            opset_version=11, 
                                            input_names=['images'],
                                            do_constant_folding=True,
                                            output_names=['classes', 'boxes'] if y is None else ['output'])
                                     ##原代码verbose=False修改为True，
                                            opset_version=12修改为11，
                                     添加参数do_constant_folding=True。
            

      4).  运行脚本：
            
        
            python3.7 models/export.py --weights ./yolov3.pt --img 416 --batch 1

        

      参数介绍：                                                         
          --weights：权重模型文件。                                   
          --img：图片大小。                                 
          --batch：batchsize大小。                              
        运行成功后，在当前目录生成yolov3.onnx模型文件。                                                 
        说明：models/export.py为yolov3官方github代码仓提供。 

    c.使用ATC工具将ONNX模型转OM模型。
    
     1).  配置环境变量。
                                     
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         
       说明：该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。

       详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

     2).  执行命令查看芯片名称型号（$\{chip\_name\}）。
        
        npu-smi info
         #该设备芯片名(${chip_name}=Ascend310P3)
         回显如下：
         +--------------------------------------------------------------------------------------------+
         | npu-smi 22.0.0                       Version: 22.0.2                                       |
         +-------------------+-----------------+------------------------------------------------------+
         | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
         | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
         +===================+=================+======================================================+
         | 0       310P3     | OK              | 16.5         55                0    / 0              |
         | 0       0         | 0000:5E:00.0    | 0            931  / 21534                            |
         +===================+=================+======================================================+
               

     3).  执行ATC命令。
         
         atc --model=yolov3.onnx 
             --framework=5 
             --output=yolov3_bs1 
             --input_format=NCHW 
             --log=info 
             --soc_version=${chip_name} 
             --input_shape="images:1,3,416,416" 
             --out_nodes="Reshape_219:0;Reshape_203:0;Reshape_187:0"
         
       说明：out_nodes为onnx模型输出节点，可能会因为pytorch版本的不同和github源码的改动导致变化，需要使用者参考本模型压缩包中提供的onnx模型和上述atc命令做一定修改，保证输出节点的位置和顺序正确。

       参数说明：

       --model：为ONNX模型文件。                                  
       --framework：5代表ONNX模型。                                   
       --output：输出的OM模型。                                               
       --input_format：输入数据的格式。                            
       --log：日志等级。                                        
       --soc_version：部署芯片类型。                                
       --input_shape：输入数据的shape。                    
       --out_nodes：输出节点名称。                        

       运行成功后生成yolov3_bs1.om文件。

2. 开始推理验证

   a. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   b. 执行推理。

   运行 ais_infer 脚本。

   
       mkdir ais_infer_result
       python3 ais_infer.py --ais_infer_path ${ais_infer_path} 
                            --model yolov3_bs1.om 
                            --input yolov3_bin 
                            --batchsize=1 
                            --output ais_infer_result
   
      推理后的输出默认在当前目录result下。

      参数说明:
          --ais_infer_path：ais-infer推理脚本`ais_infer.py`所在路径，如“./tools/ais-bench_workload/tool/ais_infer”。

      >**说明：**
      >执行ais-infer工具请选择与运行环境架构相同的命令。参数详情请参见[《ais_infer 推理工具使用文档》](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)

   c. 模型后处理。
   
   解析输出特征图。
   解析ais_infer输出文件，经过阈值过滤，nms，坐标转换等输出坐标信息和类别信息txt文件。
       
        
            python3.7 bin_to_predict_yolo_pytorch.py  
                --bin_data_path result/dumpOutput_device0/  
                --det_results_path  detection-results/ 
                --origin_jpg_path val2014/ 
                --coco_class_names coco2014.names 
                --model_type yolov3 --net_input_size 416
        

    参数说明：

    --bin_data_path：benchmark的输出路径。                                            
    --det_results_path：解析后的txt文件路径，若不存在则创建。                                                    
    --origin_jpg_path：原始图片路径。                                                
    --coco_class_name：coco数据集类型信息。                                            
    --model_type：默认为yolov5，这里选择yolov3。                                                    
    --net_input_size 416：网络输入大小，默认为640。

   d.精度验证。
   
   YOLOv3指标采用mAP0.5值，执行命令获取精度。
   
   ```
    python3 map_calculate.py --label_path  ./ground-truth-split  --npu_txt_path ./detection-results -na -np
   ```

    参数说明：

    --label_path：coco数据集标签。                                                                                    
    --npu_txt_path：上一步解析的txt文件路径。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集       |  精度 | 性能 |
| -------- | ---------- | ------------ | ------- | ---- |
| 310P3    | 1          | coco2014 | 59.28% | throughputrate:418.0131 |
