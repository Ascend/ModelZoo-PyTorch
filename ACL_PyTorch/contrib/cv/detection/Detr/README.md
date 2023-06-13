# Detr模型-推理指导

- [概述](##概述)

- [推理环境准备](##推理环境准备)

- [快速上手](##快速上手)
  
  - [获取源码](##获取源码)
  - [准备数据集](##准备数据集)
  - [模型推理](##模型推理)

- [模型推理性能](##模型推理性能)

- [配套环境](##配套环境)

## 概述

DETR是将目标检测视为一个集合预测问题（集合其实和anchors的作用类似）。由于Transformer本质上是一个序列转换的作用，因此，可以将DETR视为一个从图像序列到一个集合序列的转换过程。

- 参考论文：Carion, Nicolas, et al. "End-to-end object detection with transformers." European Conference on Computer Vision. Springer, Cham, 2020.

- 参考实现：

  ```
  url=https://github.com/facebookresearch/detr
  branch=master
  commit_id=b9048ebe86561594f1472139ec42327f00aba699
  model_name=DETR
  ```

  适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch
  tag=v.0.4.0
  code_path=ACL_PyTorch/contrib/cv/detection
  ```

  通过Git获取对应commit_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码
  cd {repository_name}              # 切换到模型的代码仓目录
  git checkout {branch/tag}         # 切换到对应分支
  git reset --hard {commit_id}      # 代码设置到对应的commit_id（可选）
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据

- 输入数据

  | 输入数据  | 数据类型 |       大小        | 数据排布格式  |
  | -------- | -------- | ---------------- | ------------ |
  | input    | RGB_FP32 |      多尺度       |      ND      |


- 输出数据

  |  输出数据    |      大小     | 数据类型  | 数据排布格式  |
  |  --------   |  -----------  | -------- | ------------ |
  | pred_boxes  |  1 x 100 x 92 | FLOAT32  |    NCHW      |
  | Pred_logits |  1 x 100 x 4  | FLOAT32  |    NCHW      |


## 推理环境准备[所有版本]

- 该模型需要以下插件与驱动。
  
  **表 1**  版本配套表

|  配套    |  版本    |环境准备指导                
| -------- | ------- |-------------
|固件与驱动 | 22.0.2  |[Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies)
| CANN     | 5.1.RC2 |-
| PyTorch  | 1.5.0   |-


## 快速上手

## 获取源码

1. 获取源码。

   ```
   上传源码包到服务器任意目录并解压（如：/home/HwHiAiUser）。 
   ├── detr.patch               #修改detr补丁
   ├── detr_excute_omval.py     #执行om推理脚本
   ├── detr_FPS.py              #计算推理性能脚本
   ├── LICENSE            
   ├── modelzoo_level.txt         
   ├── detr_onnx2om.py          #onnx转om脚本
   ├── detr_postprocess.py      #后处理脚本
   ├── detr_preprocess.py       #前处理脚本
   ├── detr_pth2onnx.py         #pth转onnx脚本
   ├── README.md                #readme文档
   ├── requirements.txt         #所需环境依赖
   └── transformer.py           #图片前处理
   ```

2. 获取开源代码仓。

   ```
   在已下载的源码包根目录下，执行如下命令。
   git clone https://github.com/facebookresearch/detr.git
   cd detr
   git checkout b9048ebe86561594f1472139ec42327f00aba699
   修改patch文件第298行，将***替换为https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
   patch -p1 < ../detr.patch
   cd ..
   ```

3. 安装依赖。

   ```
   pip3 install -r requirements.txt
   pip3 install pycocotools==2.0.3
   ```

## 准备数据集

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   
   本模型支持coco val 5000张图片的验证集。请用户根据代码仓readme获取数据集，上传数据集到代码仓目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到coco val2017.zip验证集及“instances_val2017.json”数据标签。
   
   [数据集链接](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
   
   数据目录结构请参考：
   ```
   coco_data    
   ├── annotations    
   └── val2017
   ```

2. 数据预处理。
   
   数据预处理将原始数据集转换为模型输入的数据。
   执行“detr_preprocess.py”脚本，完成预处理。

   ```
   python3.7 detr_preprocess.py --datasets=coco_data/val2017 --img_file=img_file --mask_file=mask_file
   ```

   - --datasets：原始数据验证集（.jpeg）所在路径。
   - --img_file：输出的二进制文件（.bin）所在路径。
   - --mask_file：输出的二进制文件mask bin所在路径。

   每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“img_file”和“mask_file”二进制文件夹。

## 模型推理

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
   
      从源码包中获取权重文件：[detr.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/22.1.30/ATC%20Detr%28FP16%29%20from%20Pytorch%20-%20Ascend310.zip)，将权重文件放入model文件夹。

   2. 导出onnx文件。

      1. 使用“detr.pth”导出onnx文件。

         运行detr_pth2onnx.py脚本。

         ```
         python3.7 detr_pth2onnx.py --batch_size=1
         ```

         获得“detr_bs1.onnx”文件。(本模型只支持bs1与bs4)


   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 (推理)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称(${chip_name})。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3(自行替换)
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
         mkdir auto_om
         python3.7 detr_onnx2om.py --batch_size=1 --auto_tune=False --soc_version=${chip_name}
         ```

         - 参数说明：

           - --batch_size：批大小，即1次迭代所使用的样本量。
           - --auto_tune：模型优化参数。

           运行成功后生成“auto_om”模型文件夹


2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。   


   b.  执行推理。

      ```
      mkdir result
      python3.7 detr_excute_omval.py --ais_path=ais_infer.py --img_path=img_file --mask_path=mask_file --out_put=out_put --result=result --batch_size=1 > bs1_time.log
      ```

      -   参数说明：
          -   --ais_path:ais_bench推理工具推理文件路径 
          -   --img_path:前处理的图片文件路径 
          -   --mask_path:前处理的mask文件路径 
          -   --out_put:ais_infer推理数据输出路径
          -   --result:推理数据最终汇总路径
          -   --batch_size:batch大小，可选1或4    

      执行该脚本,推理结果路径最终在result目录下，并生成推理info日志文件

   c.  精度验证。

      调用“detr_postprocess.py”脚本。

      ```
      export PYTHONPATH=usr/local/detr
      python3.7 detr_postprocess.py --coco_path=coco_data --result=result
      ```
      - usr/local/detr:源码库路径
      - --coco_path：数据集路径。
      - --result：om推理出的数据存放路径。

   d.  性能验证。

      ```
      python3.7 detr_FPS.py --log_path=bs1_time.log --batch_size=1
      ```

      解析推理日志文件，计算性能数据

      - --log_path：推理info日志文件路径


# 模型推理性能&精度

###性能

|   Model   |   Batch Size    |  310P (FPS/Card) |
| --------- | --------------- |  --------------  |
|   Detr    |        1        |     63.7531      |
|   Detr    |        4        |     60.3396      |
| 最优batch |                 |     63.7531      |

###精度

- Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.416
- Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.620
- Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.440
- Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.192
- Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.455
- Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.613
- Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.330
- Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.529
- Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.570
- Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.305
- Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.626
- Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.805

  map=41.6>42.0*0.99

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md