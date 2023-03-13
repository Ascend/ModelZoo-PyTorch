# Cascade_Mask_RCNN_UniFormer模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ******




# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

UniFormer（Unified transFormer）在arxiv中引入（更多细节可以在arxiv中找到），它可以以简洁的转换器格式无缝集成卷积和自我注意的优点。我们在浅层中采用局部MHRA来大幅减轻计算负担，并在深层采用全局MHRA来学习全局令牌关系。




- 参考实现：

  ```
  url=https://github.com/Sense-X/UniFormer.git
  commit_id=e8024703bffb89cb7c7d09e0d774a0d2a9f96c25
  code_path=contrib/cv/segmentation/Cascade_Mask_RCNN_UniFormer
  model_name=uniformer
  ```
  
 


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 800 x 1216 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | bboxes  | FLOAT32  | 100 x 5 | ND           |
  | labels  | INT64       | 100 x 1 | ND           |
  | mask_pred | FLOAT32 |  100 x 1 x 28 x 28 | NCHW |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 5.1.RC2 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
                                               



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 安装依赖。

   ```
   pip install -r requirements.txt     
   ```

2. 获取源码。
    1. 安装开源仓
   ```
   git clone -b main https://github.com/Sense-X/UniFormer.git
   cd UniFormer
   git reset e8024703bffb89cb7c7d09e0d774a0d2a9f96c25 --hard
   ```
    2. 修改模型
   ```
   patch -p1 < ../uniformer.patch
   cd object_detection
   pip install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html
   pip install -v -e .   
   cd ../..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。
   本模型已在coco 2017数据集上验证过精度。推理数据集采用coco_val_2017，请用户自行获取coco_val_2017数据集。将instances_val2017.json文件和val2017文件夹按照如下目录结构上传并解压数据集到服务器任意目录。
    最终，数据的目录结构如下：
   ```
   ├── coco
       ├── val2017   
       ├── annotations
            ├──instances_val2017.json
         

   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。


   执行uniformer_preprocess.py脚本，完成预处理。

   ```
   python uniformer_preprocess.py --image_src_path=./coco/val2017 --bin_file_path=./val2017_bin
   ```
   - 参数说明：
      -  --image_src_path：数据集路径。
      -  --bin_file_path：预处理后的数据文件的相对路径。

    
    运行成功后，会在当前目录下生成二进制文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       [cascade_mask_rcnn_3x_ms_hybrid_base.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Cascade_Mask_Rcnn_UniFormer/PTH/cascade_mask_rcnn_3x_ms_hybrid_base.pth)

   2. 导出onnx文件。

      1. 使用tools/deployment/pytorch2onnx.py导出onnx文件。

      
        进入object_detection文件中运行“tools/deployment/pytorch2onnx.py”脚本。

         ```
         python tools/deployment/pytorch2onnx.py  exp/cascade_mask_rcnn_3x_ms_hybrid_base/config.py    
         ../../cascade_mask_rcnn_3x_ms_hybrid_base.pth     --input-img demo/demo.jpg     --output-file 
         ../../uniformer_bs1.onnx --shape 800 1216

         ```
         - 参数说明：
            -  --input-img : 输入图片样例
            -  --output-file: 输出onnx模型
            -  --shape: 模型高宽

         获得uniformer_bs1.onnx文件。



   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

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

         ```
            atc --framework=5\ 
                 --model=./uniformer_bs1.onnx\ 
                 --output=./uniformer_bs1\ 
                 --input_format=NCHW\ 
                 --input_shape="input:1,3,800,1216"\ 
                 --log=error\
                 --soc_version=Ascend${ChipName}
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

        运行成功后生成uniformer_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        
    ```
    python -m ais_bench --model ${om_path}/uniformer_bs1.om\  
                                  --input ./val2017_bin/\ 
                                  --output ./\ 
                                  --batchsize 1\
                                  --outfmt BIN\
                                  --output_dirname result
    ```

    - 参数说明：

      - --model: OM模型路径。
      - --input: 存放预处理bin文件的目录路径
      - --output: 存放推理结果的目录路径
      - --batchsize：每次输入模型的样本数
      - --outfmt: 推理结果数据的格式
      - --output_dirname: 输出结果子目录
        推理后的输出默认在当前目录result下。


   3. 精度验证。

      调用“uniformer_postprocess.py”评测模型的精度。

    ```
    python uniformer_postprocess.py --ann_file_path=./coco/annotations/instances_val2017.json  --bin_file_path=result

    ```
    - 参数说明：

      - --ann_file_path：数据集路径
      - --bin_file_path：生成推理结果文件。
    
   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

    ```
    python _-m ais_bench --model ${om_path}/uniformer_bs1.om --loop 100 --batchsize 1
    ```

    - 参数说明：

      - --model: om模型
      - --batchsize: 每次输入模型样本数
      - --loop: 循环次数    



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

1. 精度对比

    | Model       | batchsize | Accuracy | 
    | ----------- | --------- | -------- |
    | Cascade_Mask_RCNN_Uniformer | 1       | bbox_mAP_50=72 segm_mAP_50=69.6 |

2. 性能对比

    | batchsize | 310 性能 | 310P 性能 | 
    | ---- | ---- | ---- |
    | 1 | 1.92  |3.01  |