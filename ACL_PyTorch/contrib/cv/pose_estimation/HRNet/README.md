# HRNet模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)


  ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

高解析网络（HRNet）在整个过程中保持高解析呈现。该网络有两个关键特性：1. 并行连接高到低分辨率的卷积流；2. 在决议之间反复交换信息。优点是得到的呈现在语义上更丰富，空间上更精确。HRNet在广泛的应用中具有优越性，包括人体姿态估计、语义分割和对象检测，表明HRNet是解决计算机视觉问题强有力的工具。


- 参考论文：[Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/pdf/1908.07919.pdf)
- 参考实现：

  ```
  url=https://github.com/HRNet/HRNet-Image-Classification.git
  branch=master
  model_name=HRNet-W18-C
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
  | input    | RGB_FP32 | batchsize x 3 x 224 x 224 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 1000 | FLOAT32  | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| onnx                                                         | 1.9.0   | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/HRNet/HRNet-Image-Classification.git
   cd HRNet-Image-Classification
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。

      数据目录结构请参考：
   ```
   ├──ImageNet
    ├──ILSVRC2012_img_val
    ├──val_label.txt
   ```

2. 数据预处理。\(请拆分sh脚本，将命令分开填写\)

   数据预处理将原始数据集转换为模型输入的数据。

   执行imagenet_torch_preprocess.py脚本，完成预处理。

   ```
   python3.7 imagenet_torch_preprocess.py hrnet ${raw_dataset_path}/ILSVRC2012_img_val ${dataset_path}
   ```
   - 参数说明：
     - hrnet：模型名称
     - ${raw_dataset_path}/ILSVRC2012_img_val：原始数据集路径
     - ${dataset_path}:经处理数据集输出路径

   每个图像对应生成一个二进制文件。运行成功后，在输出路径下生成二进制文件。



## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

       下载链接：

		 [oneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw)

       [pan.baidu.com (AccessCode:r5xn)](https://pan.baidu.com/s/1Px_g1E2BLVRkKC5t-b-R5Q)

   1. 导出onnx文件。

      1. 使用hrnet_pth2onnx.py导出onnx文件。

         运行hrnet_pth2onnx.py脚本。

         ```
          python3.7 hrnet_pth2onnx.py --cfg ./HRNet-Image-Classification/experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --input hrnet_w18.pth --output hrnet_w18.onnx
         ```

         - 参数说明：

            -   --cfg：模型配置。
            -   --input：权重文件。
            -   --output：ONNX模型文件。

         获得hrnet_w18.onnx文件。

      
   2. 使用ATC工具将ONNX模型转OM模型。

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
         ```
          atc --framework=5 --model=${onnx_model_path} --input_format=NCHW --input_shape="image:${batch_size},3,224,224" --output=hrnet_bs${batch_size} --log=debug --soc_version=${chip_name}
         ```

         - 参数说明：

           -   --model：ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。

           运行成功后生成hrnet_bs${batch_size}.om模型文件。



2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)代码仓，根据readme文档进行工具安装。  

   b.  执行推理。

      ```
       python3.7 -m ais_bench --model ${om_model_path} --input ${dataset_path} --output ${result_path} --outfmt TXT --batchsize ${batch_size} 
      ```

      - 参数说明：

        -   --model：om文件路径。
        -   --input：预处理完的数据集文件夹。
        -   --output：推理结果保存地址。
        -   --outfmt：推理结果保存格式。
        -   --batchsize：样本批量大小。

      推理后的输出默认在当前目录result下。


   c.  精度验证。

      调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

      ```
       python3.7 imagenet_acc_eval.py ${result_path} ${val_label_path}/val_label.txt ${output_path} result.json
      ```
	  
     - 参数说明：

       -   ${result_path}：生成推理结果所在路径。 
       -   val_label.txt：标签数据。
       -   ${output_path}：结果文件输出路径。
       -   result.json：生成结果文件。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| batchsize | 1       | 4       | 8       | 16      | 32      | 64      |
|-----------|---------|---------|---------|---------|---------|---------|
| 310       | 613.47  | 926.73  | 1033.33 | 1122.41 | 1136.74 | 1138.70 |
| 310P      | 600.5  | 1446.45 | 1636.7 | 2036.41 | 1533.22 | 1205.28 |

精度参考下列数据。

|      | top1   | top5   |
|------|--------|--------|
| 310  | 76.46% | 93.14% |
| 310P | 76.45% | 93.14% |