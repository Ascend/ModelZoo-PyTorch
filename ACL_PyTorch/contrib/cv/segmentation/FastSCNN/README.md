# FastSCNN模型-推理指导


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

FastSCNN：快速分割卷积神经网络（Fast-SCNN），一种基于高分辨率图像数据（1024x2048px）的实时语义分割模型，适合在低内存嵌入式设备上高效计算。


- 参考实现：

  ```
  url=https://github.com/LikeLy-Journey/SegmenTron
  branch=master 
  commit_id=4bc605eedde7d680314f63d329277b73f83b1c5f
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize * 3 *1024 * 2048 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | -------- | -------- | ----- | ------------ |
  | output1  | RGB_FP16  | batchsize * 19 * 1024 * 2048 | NCHW           |



# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>



  **表 1**  版本配套表

  | 配套            | 版本    | 环境准备指导    |
  | --------------- | ------- | ----------------------- |
  | 固件与驱动                                                   | 1.0.17  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
  | CANN                                                         | 6.0.RC1 | -|
  | python | 3.7.5 | - |
  | pytorch| 1.9.0 | - | 



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/LikeLy-Journey/SegmenTron        # 克隆仓库的代码
   ```

2. 由于onnx不支持AdaptiveAvgPool算子，需要使用module.patch修改module.py。 将FastSCNN目录下的module.patch放到FastSCNN/SegmenTron目录下，执行：
   
   ```
   cp module.patch ./SegmenTron
   cd ./SegmenTron
   git apply module.patch
   cd ..
   ```

3. 安装依赖。

   ```
   pip install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   本模型支持cityscapes leftImg8bit的500张验证集。用户需要自行获取leftImg8bit_trainvaltest.zip和gtFine_trainvaltest.zip数据集，解压，将两个文件夹放在/npu/opt/datasets/cityscapes/目录下。目录结构如下：

   ```
   ├──datasets
        ├──cityscapes
            ├──gtFine                   //验证集标注信息
                ├──test
                ├──train
                └──val
            ├──leftImg8bit              // 验证集文件夹
                ├──test
                ├──train
                └──val        
   ```

2. 数据预处理

    将原始数据集转换为模型输入的数据。

   ```
   python Fast_SCNN_preprocess.py --datasets_input_path /npu/opt/datasets/cityscapes --datasets_output_path ./prep_datasets
   ```
   - 参数说明：
     - --datasets_input_path：数据集的路径，必须写绝对路径。
     - --datasets_output_path：预处理结果的路径。

  


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
        
        −	该推理项目使用权重文件[fast_scnn_segmentron.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Fastscnn/PTH/fast_scnn_segmentron.pth)


   2. 导出onnx文件。

      1. 使用fast_scnn_segmentron.pth导出onnx文件，获得fast_scnn.onnx文件。

         运行Fast_SCNN_pth2onnx.py脚本。

         ```
         python Fast_SCNN_pth2onnx.py --pth_path fast_scnn_segmentron.pth --onnx_name fast_scnn
         ```
      - 参数说明：
        - --pth_path：pth文件所在的路径。
        - --onnx_name：生成的onnx文件名。



   2. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         #该设备芯片名为Ascend310P3
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

            atc --framework=5 --model=fast_scnn.onnx --output=fast_scnn_$｛bs｝ --input_format=NCHW --input_shape="image:$｛bs｝,3,1024,2048" --log=debug --soc_version=Ascend${chip_name} 
         ```

         - 参数说明：

           -   --model：ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的排布格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成fast_scnn_bs1.om模型文件。

2. 开始推理验证。

   1. 安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   2. 执行推理。

      推理支持batchsize1/2/4/8
 
        ```
         python3 -m ais_bench  --model ./fast_scnn_bs1.om --input ./prep_datasets/leftImg8bit/ --output ./result --output_dirname bs1 --outfmt BIN --batchsize 1
        ```

        -   参数说明：

             - --model：需要进行推理的om模型。
             - --input：模型需要的输入，支持bin文件和目录，若不加该参数，会自动生成都为0的数据，这里输入的是经过预处理目录下的bin文件。
             - --output：推理结果输出路径。默认会建立日期+时间的子文件夹保存输出结果，如果指定output_dirname，将保存到output_dirname的子文件夹下。
             - --output_dirname 推理结果保存的文件夹，和上个参数配合使用，推理文件保存在./output/output_dirname文件夹下。
             - --outfmt：输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”。
             - --batchsize：om模型的batch_size。
 
		推理后的输出默认在当前目录result下。

   3. 精度验证。

      调用Fast_SCNN_postprocess.py脚本与数据集标签prep_datasets/gitFine/比对，可以获得Accuracy数据，精度验证时间耗时较长请耐心等待，由于FastSCNN对输入模型的数据的顺序敏感，不同的验证顺序会产生不同的结果，所以使用log文件确定了输入数据的顺序。

      ```
       python Fast_SCNN_postprocess.py --result_bin_root  ./result/bs1/ --label_bin_root ./prep_datasets/gtFine/  --sort_log ./sort.log
      ```

      - 参数说明：
        - --result_bin_root：生成推理结果所在路径。
        - --label_bin_root：标签数据所在路径。
        - --sort_log：验证集数据文件名列表。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>
310推理支持batchsize:1/4,310P推理支持batchsize:1/4/8。

1. 调用ACL接口推理计算，性能参考下列数据。

   | 芯片型号 | Batch Size   | 数据集 | 性能 |
   | ------- | -------------|------| ----- |
   |  310    |     1        | cityscapes |10.2118|
   |  310    |     4        | cityscapes |12.8757|
   |  310P   |     1        | cityscapes |39.1736|
   |  310P   |     4        | cityscapes |32.7321|
   |  310P   |     8        | cityscapes |37.6047|




2. 精度如下。

   | 芯片型号 | Batch Size   | 数据集 | AvgmIou | AvgpixAcc  |
   | ------- | ------------- |------ | ------- | --------- |
   |  310    |     1         | cityscapes |  68.6667 | 95.3527 |
   |  310    |     4         | cityscapes |  68.6667 | 95.3527 |
   |  310P    |     1         | cityscapes |  68.6666 | 95.3526 |
   |  310P    |     8         | cityscapes |  68.6666 | 95.3526 |