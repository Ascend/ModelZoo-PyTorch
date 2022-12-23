#   SFA3D模型-推理指导

- [SFA3D模型-推理指导](#sfa3d模型-推理指导)
- [模型概述](#模型概述)
  - [输入输出数据](#输入输出数据)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [准备数据集](#准备数据集)
  - [模型推理](#模型推理)
- [模型推理性能](#模型推理性能)



# 模型概述<a name="00"></a>

SFA3D（Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds）是一个基于3D 激光雷达点云数据的超快速、高精度3D物体检测模型。GitHub源码[地址](https://github.com/maudzung/SFA3D)和技术细节[地址](https://github.com/maudzung/SFA3D/blob/master/Technical_details.md)。

- 参考实现:

  ```
  url=https://github.com/maudzung/SFA3D.git
  commit_id=5f042b9d194b63d47d740c42ad04243b02c2c26a
  model_name=fpn_resnet_18_epoch_300
  ```
  
  

## 输入输出数据<a name="00_1"></a>

- 输入数据

    | 输入数据              | 解释                                                     | 数据类型 | 大小              | 数据排布格式 |
    | :-------------------- | :------------------------------------------------------- | :------- | :---------------- | :----------- |
    | BEV（birds-eye-view） | BEV图（bin文件）按3DLiDAR 点云的高度、强度和密度进行编码 | RGB_FP32 | N x 3 x 608 x 608 | NCHW         |

- 输出数据

    | 输出数据   | 解释                                                         | 数据类型 | 大小              | 数据排布格式 |
    | :--------- | :----------------------------------------------------------- | :------- | ----------------- | ------------ |
    | hm_cen     | Heatmap(main center) ：主图目标 ``(H/S, W/S, C)``，下采样率``S=4`` ，目标类``C=3`` (含Cars, Pedestrians, Cyclists) | FLOAT32  | N x 3 x 152 x 152 | NCHW         |
    | cen_offset | Center offset：中心偏移`` (H/S, W/S, 2)``                    | FLOAT32  | N x 2 x 152 x 152 | NCHW         |
    | direction  | The heading angle (yaw) ：偏向角，估计虚部和实数分数  ``(H/S, W/S, 2)  sin(yaw)  cos(yaw)`` | FLOAT32  | N x 2 x 152 x 152 | NCHW         |
    | z_coor     | Coordinate：坐标 ``(H/S, W/S, 1)``                           | FLOAT32  | N x 1 x 152 x 152 | NCHW         |
    | dim        | Dimension：(h, w, l) `(H/S, W/S, 3)`                         | FLOAT32  | N x 3 x 152 x 152 | NCHW         |



# 推理环境准备<a name="01"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.3  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 |                                                              |
  | Python                                                       | 3.7.5   | -                                                            |
  | PyTorch                                                      | 1.6.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="1"></a>

## 获取源码<a name="1_0"></a>

1. 获取源码。

   ```
   git clone https://github.com/maudzung/SFA3D.git ./SFA3D
   cd SFA3D
   git reset --hard 5f042b9d194b63d47d740c42ad04243b02c2c26a
   ```

2. 安装依赖

   ```
   pip3 install -r requirements.txt
   ```

   

## 准备数据集<a name="1_1"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   KITTI数据集是目前国际上最大的自动驾驶场景下的计算机视觉算法评测数据集，该数据集可用于评测3D物体检测(object detection)等计算机视觉技术在车载环境下的性能，本模型label细分为car, pedestrian, cyclist 3类。

   本模型支持KITTI数据集velodyne数据的验证集，用户需自行获取数据集，将数据集放至源码SFA3D/dataset/kitti/目录下。
   所需数据集包含图片和点云数据，其中，training数据7481张，testing数据7518张（无label数据）：

   - Velodyne point clouds (29 GB) 点云数据（作为模型输入的原始数据）
   - Training labels of object data set (16 MB) 训练集标签
   - Camera calibration matrices of object data set (16MB) 相机校准矩阵
   - Left color images of object data set (12GB)  RGB图像（不作为模型输入数据，仅供可视化使用）

   数据集下载途径：KITTI数据集官网 [3D KITTI detection dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)，目前官网可能存在访问限制，可自行寻找公开资源下载。

   构造源代码和数据集目录结构如下：

   ```
   SFA3D
   
   └── checkpoints			//预训练权重文件夹
   	...
   └── dataset				//数据集
       └── kitti
           ├──ImageSets	//数据集文件名信息
           │   ├── test.txt
           │   ├── train.txt
           │   └── val.txt
           ├── training	//训练集（含验证集）
           │   ├── image_2	//(left color camera)
           │   ├── calib/
           │   ├── label_2/
           │   └── velodyne/
           └── testing		//测试集（无label数据）
           │   ├── image_2/ (left color camera)
           │   ├── calib/
           │   └── velodyne/
           └── classes_names.txt
   └── sfa/
   	...
   ...
   ```

   说明：将源码SFA3D文件夹与本篇执行脚本放置在同一目录下。

   

2. 数据预处理，将原始数据集转换为模型输入的数据。

   veloyne点云数据（bin文件）预处理，处理方法由官方github代码仓提供。执行SFA3D_preprocess.py 脚本，将原始数据集转换为模型输入的数据。

    ```
    python3 SFA3D_preprocess.py src_path input_data_save_path
    ```

    参数说明：\
   scr_path：veloyne点云数据（bin文件）文件夹路径。\
   input_data_save_path：生成的文件路径（若不存在则自动创建文件夹）。

    执行成功后input_data_save_path中生成的bin文件作为模型的输入数据。



## 模型推理<a name="1_2"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. pytorch权重文件获取

      源码中pth权重文件路径：SFA3D/checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth

      [SFA3D预训练pth权重文件](https://github.com/maudzung/SFA3D/blob/master/checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth)

      

   2. 导出onnx文件。

        将模型权重pth文件转为onnx文件。执行SFA3D_pth2onnx.py脚本：

        ```
       python3 SFA3D_pth2onnx.py --pretrained_path pth --onnx_path onnx  
       ```

   
   ​		参数说明：

   ​		--pretrained_path：pytorch权重文件，若不输入该参数，默认为当前路径。\
   ​		--onnx_path：生成的onnx模型，默认SFA3D.onnx。

   ​		运行成功后，在相应目录生成SFA3D.onnx模型文件，batchsize为动态设定。
   
   
   
   3. 使用atc工具将onnx模型转om模型。
      1. 环境变量配置

            ```
            source /usr/local/Ascend/ascend-toolkit/set_env.sh
            # 说明：该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。
            ```
      
        2. 执行命令查看芯片名称型号（$\{chip\_name\}）
      
            ```
            npu-smi info
            #该设备芯片名(${chip_name}=Ascend310P3)
            会显如下：
            +--------------------------------------------------------------------------------------------+
            | npu-smi 22.0.0                       Version: 22.0.2                                       |
            +-------------------+-----------------+------------------------------------------------------+
            | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
            | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
            +===================+=================+======================================================+
            | 0       310P3     | OK              | 16.5         55                0    / 0              |
            | 0       0         | 0000:5E:00.0    | 0            931  / 21534                            |
            +===================+=================+======================================================+
           ```
           
        2. 执行atc命令，详细工具使用方法请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。
      
            ```
            atc --framework=5
               --model=SFA3D.onnx
               --input_format=NCHW
               --input_shape="inputs:$n,3,608,608"
               --output=SFA3D_bs${n}
               --log=error
               --soc_version=${chip_name} 
            ```
            
            参数说明：\
               n为batchsize设定。\
              --framework：5代表ONNX模型。\
              --model：ONNX模型文件。\
              --output：输出的OM模型。\
              --input_format：输入数据的格式。\
              --log：日志等级。\
              --soc_version：部署芯片类型。\
              --input_shape：输入数据的shape，该网络模型输入节点名称为"inputs"。

              运行成功后在指定路径生成SFA3D_bs${n}.om文件。
      
2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请点击本链接进行安装ais_bench推理工具，以及查看具体使用方法(https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)

   2. 执行推理。

      ```
      mkdir ais_infer_result
      python3 -m ais_bench --model SFA3D_bs${n}.om 
                          --input ${input_data_save_path} 
                          --batchsize=${n}
                          --output ais_infer_result
      ```

      参数说明:\
      n为batchsize设定。\
      --model：待推理的om模型。\
      --input：模型输入，支持bin文件和目录，此例为数据文件夹路径。\
      --output：推理结果输出路径。

      推理后样本的输出在当前目录的ais_infer_result文件夹下，默认会建立日期+时间的子文件夹保存输出结果。

      **说明：** 执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见。

      

3. 精度验证。

   本篇用模型原代码仓提供的loss计算方法评估。SFA3D模型推理有5个输出文件（hm_cen, cen_offset, direction, z, dim）；解析ais_bench推理结果，调用脚本与标签文件的targets比对，计算loss。

   - 对于主图目标（hm_cen）使用 focal_loss
   - 对于中心偏移和偏向角（cen_offset, direction）使用 l1_loss
   - 对于坐标和长宽高维度（z, dim）使用 balanced l1_loss

   1. 离线推理loss统计。

      执行SFA3D_postprocess.py脚本，结果打印至当前目录SFA3D_om_losses.txt中。

        ```
        python3 SFA3D_postprocess.py --dataset_dir kitti_path --result_path infer_result > SFA3D_om_losses.txt
        ```

        参数说明：\
        --dataset_dir：KITTI数据集路径，默认路径为 "./SFA3D/dataset/kitti/" 目录。\
        --result_path：推理结果路径，默认为ais_infer_result目录下文件，例如 "./ais_infer_result/dumpdata_outputs/ "。
      
   2. 开源模型loss统计。

      运行脚本SFA3D_val_losses.py，结果打印至当前目录SFA3D_pth_losses.txt中。

      ```
      python3 SFA3D_val_losses.py > SFA3D_pth_losses.txt
      ```

      说明：该脚本中路径参数可修改，默认为脚本与源码SFA3D文件夹处于同一目录下；dataset_dir为KITTI数据集路径，pretrained_path为pth权重文件路径。
   
   

   - loss值对比
   
       | loss_avg | total  | hm_cen | cen_offset | dim    | direction | z      |
       | -------- | ------ | ------ | ---------- | ------ | --------- | ------ |
       | om模型   | 0.6038 | 0.4364 | 0.1056     | 0.0182 | 0.0367    | 0.0070 |
       | 开源模型 | 0.6038 | 0.4364 | 0.1056     | 0.0182 | 0.0367    | 0.0070 |
    
       此处展示bs1的loss值，本模型bs1和最优bs的loss值无差别。
     
       om离线模型推理loss值与该模型github源码仓推理得到的loss值一致，故精度达标。




4. 性能验证。

   纯推理数据，测试时确保服务器上无其他进程。

   1. ais_bench纯推理。
      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

      ```
      python3.7 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
      ```

   2. trt纯推理。
      使用trtexec工具推理。TensorRT版本：7.2.3.4，cuda版本：11.0，cudnn版本：8.2

      ```
      trtexec --onnx=SFA3D.onnx --shapes=inputs:nx3x608x608 --fp16 --threads --iterations=N
      ```

      参数说明：\
      n为batchsize设定。\
      --shapes：为动态设定batchsize或shape的ONNX模型设置输入shape；若ONNX为静态模型则无需设置。\
      --iterations：设置推理迭代次数（默认值为10）




# 模型推理性能<a name="2"></a>

调用ACL接口推理计算，性能参考下列数据。

| bs     | 310P   | T4     | 310P/T4 |
| ------ | ------ | ------ | ------- |
| 1      | 423.97 | 221.78 | 1.91    |
| 4      | 426.74 | 260.80 | 1.64    |
| 8      | 401.30 | 272.32 | 1.47    |
| 16     | 395.61 | 274.30 | 1.44    |
| 32     | 372.34 | 274.29 | 1.36    |
| 64     | 375.20 | 276.37 | 1.36    |
| 最优bs | 426.74 | 276.37 | 1.54    |




