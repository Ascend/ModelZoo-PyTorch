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

| 配套           | 版本               | 环境准备指导                                                 |
|--------------|------------------| ------------------------------------------------------------ |
| 固件与驱动        | 23.0.RC1         | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN         | 7.0.RC1.alpha003 | -                                                            |
| Python       | 3.9.11           | -                                                            |
| PyTorch      | 2.0.1            | -                                                            |
| Torch_AIE    | 6.3.rc2          | \                                                            |
| 芯片类型         | Ascend310P3      | \                                                            |
                                               


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
   patch -p1 < ../Cascade_RCNN-detectron2.patch
   ```
   3. 修改读取数据集路径
   ```
   vi detectron2/detectron2/data/datasets/builtin.py
   ```
   修改如下代码：将数据集路径改为实际数据集路径，即包含coco文件夹的目录路径
   ```
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

   使用PyTorch将模型权重文件.pth转换为.pt文件

   1. 获取权重文件。

      下载[权重文件](https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_R_50_FPN_3x/144998488/model_final_480dd8.pkl)。
   2. 编译自定义算子，使用如下命令
      ```
      cd Cascade_RCNN_ops
      export PATH=/path/to/torch:${PATH}  # 此处/path/to/torch填写上述环境安装后的torch1.11.0的路径
      bash build.sh
      cd ..
      ```

   3. 导出ts文件。

      1. 使用detectron2/tools/deploy/export_model.py导出ts文件（需要提前在export_model.py134行处加入上述命令编出的so路径）

         ```
         python detectron2/tools/deploy/export_model.py --config-file detectron2/configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml --output ./ \
         --export-method tracing --format torchscript MODEL.WEIGHTS model_final_480dd8.pkl MODEL.DEVICE cpu
         ```
         - 参数说明：
            -  --config-file : 配置文件
            -  --output: 输出模型
          
         获得model.onnx文件,模型只支持bs1。

   3. 转换AIE模型。
       ```
         python cascadercnn_infer.py \
         --torch_model  ./output/model_torch.ts \
         --aie_model ./output/mmdet_aie.ts \
         --device 0 \
         --only_compile
       ```
      参数说明：
      - --torch_model: ts文件
      - --aie_model: 保存的aie模型名称
      - --device: 指定设备
      - --only_compile:仅导出模型

2. 开始推理验证。

   1. 运行下述脚本进行模型推理并保存推理结果。

      ```
      python cascadercnn_infer.py \
        --torch_model  ./torch.ts \
        --aie_model ./aie.ts \
        --infer_data ./val2017_bin \
        --infer_result ./result \
        --device 0 \
        --precision
      ```

      - 参数说明：
        - --torch_model: 序列化TorchScript模型保存路径
        - --aie_model: 模型编译转换后的保存路径
        - --infer_data: 模型推理数据路径
        - --infer_result: 推理结果保存路径
        - --device: 设备号
        - --precision: 若设置则保存推理结果

      推理后的输出默认在当前目录result下。

   2. 精度验证。

      运行get_info.py,生成图片数据文件
      ```
      python get_info.py jpg ./coco/val2017 cascadercnn_jpeg.info
      ```
      - 参数说明：

         - --第一个参数：原始数据集
         - --第二个参数：图片数据信息

      调用“cascadercnn_postprocess.py”评测模型的精度。

      ```
      python cascadercnn_postprocess.py \
       --bin_data_path ./result \
       --test_annotation ./val2017.info \
       --det_results_path ./precision_result

      ```
      - 参数说明：
          - --bin_data_path: 模型推理的结果路径
          - --test_annotation: 推理数据的标签
          - --det_results_path: 精度测试结果的保存路径
    
   3. 性能验证。

      运行下述脚本进行性能验证

      ```
      python cascadercnn_infer.py \
        --torch_model  ./output/model_torch.ts \
        --aie_model ./output/mmdet_aie.ts \
        --device 0 \
        --infer_with_zeros
      ```

      - 参数说明：

          - --torch_model: 序列化TorchScript模型保存路径
          - --aie_model: 模型编译转换后的保存路径
          - --device: 设备号
          - --infer_with_zeros: 与ais_bench对比性能 

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

性能参考下列数据。

1. 精度对比

    | Model       | batchsize | Accuracy | 
    | ----------- | --------- | -------- |
    | Cascadercnn | 1       | bbox_mAP = 43.9 |

2. 性能对比

    | batchsize | 310P 性能 | 
    | ---- |---------|
    | 1 | 11.5    |