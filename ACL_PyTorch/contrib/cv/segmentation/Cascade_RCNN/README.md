# Cascade_RCNN-detectron2模型-推理指导


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

在目标检测中，需要一个交并比(IOU)阈值来定义物体正负标签。使用低IOU阈值(例如0.5)训练的目标检测器通常会产生噪声检测。然而，随着IOU阈值的增加，检测性能趋于下降。影响这一结果的主要因素有两个：1)训练过程中由于正样本呈指数级消失而导致的过度拟合；2)检测器为最优的IOU与输入假设的IOU之间的推断时间不匹配。针对这些问题，提出了一种多级目标检测体系结构-级联R-CNN.它由一系列随着IOU阈值的提高而训练的探测器组成，以便对接近的假阳性有更多的选择性。探测器是分阶段训练的，利用观察到的探测器输出是训练下一个高质量探测器的良好分布。逐步改进的假设的重采样保证了所有探测器都有一组等效尺寸的正的例子，从而减少了过拟合问题。同样的级联程序应用于推理，使假设与每个阶段的检测器质量之间能够更紧密地匹配。Cascade R-CNN的一个简单实现显示，在具有挑战性的COCO数据集上，它超过了所有的单模型对象检测器。实验还表明，Cascade R-CNN在检测器体系结构中具有广泛的适用性，独立于基线检测器强度获得了一致的增益。




- 参考实现：

  ```
  url=https://github.com/facebookresearch/detectron2.git
  commit_id=13afb035142734a309b20634dadbba0504d7eefe
  code_path=contrib/cv/segmentation/Cascade_RCNN
  model_name=detectron2
  ```
  
 


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 1344 x 1244 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | 100 x 4 | ND           |
  | output2  | FLOAT32       | 100 x 1 | ND           |
  | output3 | INT64 |  100 x 1  | ND |
  | output4 | FLOAT32 | 100 x 80 x 28 x 28 | NCHW|

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动   

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17（NPU驱动固件版本为6.0.RC1） | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
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
   git clone https://github.com/facebookresearch/detectron2
   cd detectron2
   git reset --hard 13afb035142734a309b20634dadbba0504d7eefe
   python -m pip install -e .

   ```
    2. 修改模型
   ```
   patch -p1 < ../cascadercnn_detectron2.diff
   ```
    3. 屏蔽torch.onnx model_check相关代码
   ```
   cd ..
   mv export.patch ./detectron2/tools/deploy
   ```
   进入/detectron2/tools/deploy中
   ```
   patch -p 1 export_model.py export.patch
   ```
    4. 修改读取数据集路径
   ```
   vi detectron2/detectron2/data/datasets/builtin.py
   if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "./datasets/")//修改为数据集实际路径
    register_all_coco(_root)
    register_all_lvis(_root)
    register_all_cityscapes(_root)
    register_all_cityscapes_panoptic(_root)
    register_all_pascal_voc(_root)
    register_all_ade20k(_root)
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


   执行cascadercnn_preprocess.py脚本，完成预处理。

   ```
   python cascadercnn_preprocess.py --image_src_path=./coco/val2017 --bin_file_path=val2017_bin --model_input_height=1344 --model_input_width=1344
   ```
   - 参数说明：
      -  --image_src_path：数据集路径。
      -  --bin_file_path：预处理后的数据文件的相对路径。
      -  --model_input_height: 图片高
      -  --model_input_width： 图片宽
    
    运行成功后，会在当前目录下生成二进制文件。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       该推理项目使用源码包中的权重文件（cascadercnn_detectron2.pkl）。

   2. 导出onnx文件。

      1. 使用detectron2/tools/deploy/export_model.py导出onnx文件。

      
  

         ```
         python detectron2/tools/deploy/export_model.py --config-file detectron2/configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml --output ./ --export- 
         method tracing --format onnx MODEL.WEIGHTS cascadercnn_detectron2.pkl MODEL.DEVICE cpu

         ```
         - 参数说明：
            -  --config-file : 配置文件
            -  --output: 输出onnx模型
          

         获得model.onnx文件,模型只支持bs1。



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
                 --model=./model.onnx\ 
                 --output=./cascadercnn_detectron2_npu\ 
                 --input_format=NCHW\ 
                 --input_shape="0:1,3,1344,1344"\ 
                 --log=debug\
                 --out_nodes="Cast_1853:0;Gather_1856:0;Reshape_1847:0;Slice_1886:0"\
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
           -   --out_nodes: 输出节点

        运行成功后生成cascadercnn_detectron2_npu.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

        
    ```
    python3 -m ais_bench --model ./cascadercnn_detectron2_npu.om\  
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

      运行get_info.py,生成图片数据文件
    ```
    python get_info.py jpg ./coco/val2017 cascadercnn_jpeg.info
    ```
    - 参数说明：

      - --第一个参数：原始数据集
      - --第二个参数：图片数据信息

      调用“cascadercnn_postprocess.py”评测模型的精度。

    ```
    python cascadercnn_postprocess.py --bin_data_path=./result --test_annotation=cascadercnn_jpeg.info --det_results_path=./ret_npuinfer --net_out_num=4 
    --net_input_height=1344 --net_input_width=1344 –ifShowDetObj

    ```
    - 参数说明：

      - --bin_data_path: 推理结果。
      - --test_annotatio: 原始图片信息文件。
      - --det_results_path: 后处理输出结果。
      - --net_out_num: 网络输出个数。
      - --net_input_heigh: 网络高
      - --net_input_width：网络宽。
      - --ifShowDetObj：是否将box画在图上显示。
    
   4. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

    ```
    python3 -m ais_bench --model ./cascadercnn_detectron2_npu.om --loop 100 --batchsize 1
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
    | Cascadercnn | 1       | bbox_mAP = 44.2 |

2. 性能对比

    | batchsize | 310 性能 | 310P 性能 | 
    | ---- | ---- | ---- |
    | 1 | 4.42  |9.3  |