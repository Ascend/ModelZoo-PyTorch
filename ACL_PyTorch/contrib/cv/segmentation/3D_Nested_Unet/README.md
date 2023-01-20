# 3D_Nested_Unet模型-推理指导


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

UNet++由不同深度的U-Net组成，其解码器通过重新设计的跳接以相同的分辨率密集连接。UNet++中引入的经过重新设计的跳接在解码器节点处提供了不同比例的特征图，从而使聚合层可以决定如何将跳接中携带的各种特征图与解码器特征图融合在一起。通过以相同的分辨率密集连接组成部分U-Net的解码器，可以在UNet++中实现重新设计的跳接。
- 参考实现：[Zhou Z, Rahman Siddiquee M M, Tajbakhsh N, et al. Unet++: A nested u-net architecture for medical image segmentation[M]//Deep learning in medical image analysis and multimodal learning for clinical decision support. Springer, Cham, 2018: 3-11.](https://pubmed.ncbi.nlm.nih.gov/32613207/) 

- 参考实现：

   ```
   url=https://github.com/MrGiovanni/UNetPlusPlus.git
   branch=master
   commit_id=e145ba63862982bf1099cf2ec11d5466b434ae0b
   ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | FLOAT32  | 1 x 1000 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动 

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -                                                            |
  | Python                                                       | 3.7.5   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码

   ```
   git clone https://github.com/MrGiovanni/UNetPlusPlus.git -b master
   cd UNetPlusPlus
   git reset e145ba63862982bf1099cf2ec11d5466b434ae0b --hard

   patch -p1 < ../new.patch
   pip install -r requirements.txt
   cd pytorch
   pip install -e .
   cd ../../

   mkdir environment
   cd environment
   mkdir nnUNet_raw_data_base
   mkdir nnUNet_preprocessed
   mkdir RESULTS_FOLDER
   mkdir input output input_bins

   pwd_path=` pwd `
   export nnUNet_raw_data_base="$pwd_path/nnUNet_raw_data_base"
   export nnUNet_preprocessed="$pwd_path/nnUNet_preprocessed"
   export RESULTS_FOLDER="$pwd_path/RESULTS_FOLDER"
   mkdir -p $pwd_path/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1
   cd ..
   ```
   目录结构如下：

   ```
   ├── UNetPlusPlus
   ├── environment
         ├── nnUNet_raw_data_base
         ├── nnUNet_preprocessed
         ├── RESULTS_FOLDER
         ├── input
         ├── nnUNet_preprocessed
         ├── RESULTS_FOLDER        
   ├── 3d_nested_unet_pth2onnx.py
   ├── 3d_nested_unet_preprocess.py
   ├── 3d_nested_unet_postprocess.py
   ├── change_infer_path.py
   ├── onnx_infer.py
   ├── new.patch
   ├── License
   ├── README.md
   ├── requirements.txt
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持[Task03_Liver.tar](http://medicaldecathlon.com/)任务集。上传数据集到源码包路径下。以"$pwd_path/environment"目录结构如下：

   ```
   ├──./environment
      ├──Task03_Liver
         ├── imagesTr
         ├── labelsTr      
         └── imagesTs
   ```

2. 数据预处理，将原始数据集转换为模型输入的数据。

   依次执行命令，完成预处理（出现任何数据报错，请删除environment文件夹所有文件，重新执行）。

   1. 设置环境变量
   ```
   python3 change_infer_path.py -fp1 $pwd_path/input/ -fp2 $pwd_path/output/ -fp3 $pwd_path/
   ```

   2. 使用nnunet的脚本命令，对解压出的Task03_Liver文件夹中的数据进行数据格式转换。
   ```
   nnUNet_convert_decathlon_task -i Task03_Liver -p 8
   ```

   - 参数说明：
      - -i：任务名。
      - -p：并行cpu数（可根据环境情况自行修改）。

   3. 提取数据集的属性
   ``` 
   nnUNet_plan_and_preprocess -t 003 --verify_dataset_integrity
   ```
   转换结果将出现在nnUNet_preprocessed子文件夹中。
   > 注：通过输入free -m命令，如果系统显示的available Mem低于30000或在30000左右，则我们推荐您使用下面的命:nnUNet_plan_and_preprocess -t 003 --verify_dataset_integrity -tl 1 -tf 1

   ```
   cd $pwd_path/../UNetPlusPlus/pytorch/nnunet/inference
   python3 create_testset.py $pwd_path/input/
   cd -
   ```
   4. 将所有的.nii.gz后缀结果放置于"$pwd_path/output"下，删除任一文件，以下的推理便是针对此文件进行，然后执行3d_nested_unet_preprocess.py命令,进行预处理。

   ```
   python3 3d_nested_unet_preprocess.py --file_path $pwd_path/input_bins/
   ```
   - 参数说明：
      - --file_path：预处理后的文档。
   > 注：依次删除文件进行完成所有文件的推理。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      下载权重文件[fold_0 和 plans.pkl](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2FMrGiovanni%2FUNetPlusPlus%2Ftree%2Fmaster%2Fpytorch)，放置于"$pwd_path/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1/"文件夹下。

   2. 导出onnx文件。

      1. 使用3d_nested_unet_pth2onnx.py导出onnx文件。
         运行3d_nested_unet_pth2onnx.py脚本。

         ```
         python3 3d_nested_unet_pth2onnx.py --file_path ./nnunetplusplus.onnx
         ```

         获得nnunetplusplus.onnx文件。


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
         atc --framework=5 --model=nnunetplusplus.onnx --output=nnunetplusplus --input_format=NCDHW --input_shape="image:1,1,128,128,128" --log=error --soc_version=Ascend${chip_name}
         ```

         - 参数说明：
            - --model：为ONNX模型文件。
            - --framework：5代表ONNX模型。
            - --output：输出的OM模型。
            - --input\_format：输入数据的格式。
            - --input\_shape：输入数据的shape。
            - --log：日志级别。
            - --soc\_version：处理器型号。

           运行成功后生成./nnunetplusplus.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。

   2. 执行推理。

      ```
      mkdir  $pwd_path/result/
      python3 -m ais_bench --model=./nnunetplusplus.om --input=$pwd_path/input_bins/  --output=./$pwd_path/result/ --output_dirname=bs1 --outfmt=BIN  --batchsize=1 --device=0  
      ```

      - 参数说明：
         - --model：om文件路径。
         - --input：输入的bin文件路径。
         - --output：推理结果文件路径。
         - --outfmt：输出结果格式。
         - --device：NPU设备编号。
         - --batchsize：批大小。

      推理后的输出默认在当前目录$pwd_path/result/bs1下。


   3. 精度验证。

      1. 调用脚本进行后处理。

      ```
      python3 3d_nested_unet_postprocess.py --file_path $pwd_path/result/bs1
      ```

      - 参数说明：
         - --file_path：om推理结果的路径。
      处理一个图片的的数据存放在$pwd_path/output路径下。

      > 注：可执行以下命令进行一键式进行所有数据集的前后处理以及npu推理。
      >```
      >python3 infer.py  --environment=$pwd_path/ --interpreter=python3 --npu_interpreter="python3 -m ais_bench" --om_path=./nnunetplusplus.om --device=0
      >```
      >

      2 . 依次执行命令进行精度评测
      ```
      mkdir  $pwd_path/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/
      cp -rf $pwd_path/output/* $pwd_path/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/
      rm $pwd_path/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/*_0000.nii.gz
      nnUNet_train 3d_fullres nnUNetPlusPlusTrainerV2 003 0 --validation_only
      ```
      实验的精度将记录在$pwd_path/RESULTS_FOLDER/nnUNet/3d_fullres/Task003_Liver/nnUNetPlusPlusTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/summary.json中。
3. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：
    
      ```python
      python3 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size} --device=${device_id} --outfmt=BIN
      ```

      - 参数说明：
         - --model：om模型路径。
         - --batchsize：批大小。
         - --loop：推理循环次数。 


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | :-------: | ---------- | ---------- | --------------- |
| Ascend310P3 |   1   |   Task03_Liver  |     Liver 1_Dice (val):96.55, Liver 2_Dice (val):71.97      |    3.983    |