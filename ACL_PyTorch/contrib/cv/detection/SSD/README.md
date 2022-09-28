# SSD300模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)





# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

SSD将detection转化为regression的思路，可以一次完成目标定位与分类。该算法基于Faster RCNN中的Anchor，提出了相似的Prior box；该算法修改了传统的VGG16网络：将VGG16的FC6和FC7层转化为卷积层，去掉所有的Dropout层和FC8层。同时加入基于特征金字塔的检测方式，在不同感受野的feature map上预测目标。




- 参考实现：

  ```
   url=https://github.com/open-mmlab/mmdetection.git
   branch=master
   commit_id=a21eb25535f31634cef332b09fc27d28956fb24b
   model_name=ssd
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
  | input    | RGB_FP32 | batchsize x 3 x 300 x 300 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | boxes  | batchsize x 8732 x 4  | FLOAT32  | ND           |
  | labels  | batchsize x 8732 x 80 | FLOAT32  | ND           |
  




# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.1   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>


## 准备环境<a name="section183221994411"></a>

   1. 环境安装
      ```
      pip install -r requirements.txt
      ```

   2. mmdetection源码安装。
      ```
      git clone https://github.com/open-mmlab/mmdetection.git
      cd mmdetection
      git reset --hard a21eb25535f31634cef332b09fc27d28956fb24b
      pip install -v -e .
      ```

   3. 通过打补丁的方式修改mmdetection。
      ```
      patch -p1 < ../ssd_mmdet.diff
      ```



## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   数据集名称：coco2017

   所用到的文件：推理数据集采用coco_val_2017

   下载链接：http://images.cocodataset.org

   存放路径：/root/datasets/

   目录结构：

   ```
   ├── coco
   │    ├── val2017   
   │    ├── annotations
   │         ├──instances_val2017.json
   ```

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   将原始数据集转换为模型输入的二进制数据。执行mmdetection_coco_preprocess脚本。


   ```
   python mmdetection_coco_preprocess.py --image_folder_path /root/datasets/coco/val2017 --bin_folder_path val2017_ssd_bin
   ```

   - 参数说明：

      -   --image_folder_path：原始数据验证集（.jpg）所在路径。
      -   --bin_folder_path：输出的二进制文件（.bin）所在路径。


   每个图像对应生成一个二进制文件。


3. 生成数据集info文件。

   运行get_info.py脚本，生成图片数据info文件。
   ```
   python get_info.py jpg /root/datasets/coco/val2017 coco2017_ssd_jpg.info
   ```

   - 参数说明：

      -   第一个参数：生成的数据集文件格式。
      -   第二个参数：预处理后的数据文件相对路径。
      -   第三个参数：生成的info文件名。
   
   运行成功后，在当前目录中生成coco2017_ssd_jpg.info。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       获取经过训练的权重文件ssd300_coco_20200307-a92d2092.pth:
       ```
       wget http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth
       ```

   2. 导出onnx文件。

      使用pytorch2onnx.py导出onnx文件。

      ```
      python mmdetection/tools/pytorch2onnx.py mmdetection/configs/ssd/ssd300_coco.py ./ssd300_coco_20200307-a92d2092.pth --output-file=ssd300_coco_dynamic_bs.onnx --shape=300 --show --mean 123.675 116.28 103.53 --std 1 1 1
      ```

      - 参数说明：

         -   --output-file：为ONNX模型文件。
         -   --shape：输入的图片大小。
         -   --show：输出的OM模型。
         -   --mean：输入数据的格式。
         -   --std：输入数据的shape。

      获得ssd300_coco_dynamic_bs.onnx文件。


   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

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

         设置环境变量：
         ```
         export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
         ```

         执行atc命令
         ```
         atc --model=ssd300_coco_dynamic_bs.onnx --framework=5 --output=${om_name} --input_format=NCHW --input_shape="input:${batchsize},3,300,300" --log=debug --soc_version=Ascend${chip_name} --buffer_optimize=off_optimize --precision_mode=allow_fp32_to_fp16
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --buffer_optimize：
           -   --precision_mode：

           运行成功后生成ssd300_coco_bs8.om模型文件。



2. 开始推理验证。

   1. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

   2. 执行推理。


      ```
      python tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ${om_path} --input ./val2017_ssd_bin --batchsize ${batchsize} --output ${out_path}      
      ```

      - 参数说明：

         -   --model：为.OM模型文件的路径。
         -   --input：转换之后的二进制数据集路径。
         -   --batchsize：batch维度大小，与输入的.OM模型文件的batch维度一致。
         -    --output：模型推理结果存放的路径。

      上述命令将会在 ${output} 所在目录创建一个以时间命名的文件夹来存放推理结果。

   2. 精度验证。

      调用coco_eval.py评测map精度：

      ```
      python mmdetection_coco_postprocess.py --bin_data_path=${infer_result_path} --score_threshold=0.02 --test_annotation=coco2017_ssd_jpg.info --nms_pre 200 --det_results_path ${det_path}
      python txt_to_json.py --npu_txt_path ${det_path} 
      python coco_eval.py --ground_truth /root/datasets/coco/annotations/instances_val2017.json
      ```

      - 参数说明：

         -   --bin_data_path：为推理结果存放的路径。
         -   --score_threshold：得分阈值。
         -   --test_annotation：原始图片信息文件。
         -   --nms_pre：每张图片获取框数量的阈值。
         -   --det_results_path：后处理输出路径。
         -   --npu_txt_path：后处理输出路径。
         -   --ground_truth：instances_val2017.json文件路径。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。


|   |  mAP |
|---|---|
| 310精度  |  mAP=25.4 |
| 310P精度  |  mAP=25.4 |
| 性能  |  337.01 |


|   Throughput      | 310      | 310P     | T4       | 310P/310    | 310P/T4     |   
|---------|----------|----------|----------|-------------|-------------|
| bs1     | 179.194  | 298.5514 | 250.8491 | 1.666079221 | 1.190163329 |   
| bs4     | 207.596  | 337.0112 | 310.1569 | 1.623399295 | 1.086582952 |   
| bs8     | 211.7312 | 323.5662 | 332.0797 | 1.528193294 | 0.974363082 |   
| bs16    | 211.288  | 318.1392 | 352.4384 | 1.505713528 | 0.902680298 |   
| bs32    | 200.2948 | 318.7303 | 348.0656 | 1.591305915 | 0.915719048 |  
| bs64    | 196.4192 | 313.0790 | 370.7415 | 1.593932772 | 0.844467102 |   
| 最优batch | 211.7312 | 337.0112 | 370.7415 | 1.591693619 | 0.909019357 | 