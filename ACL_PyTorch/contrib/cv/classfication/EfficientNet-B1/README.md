# EfficientNet-B1模型PyTorch离线推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)



- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

EfficientNet是图像分类网络，在ImageNet上性能优异，并且在常用迁移学习数据集上达到了相当不错的准确率，参数量也大大减少，说明其具备良好的迁移能力，且能够显著提升模型效果。


- 参考实现：

  ```
<<<<<<< HEAD
  url=https://github.com/facebookresearch/pycls
=======
  url=https://github.com/rwightman/pytorch-image-models
>>>>>>> b50bfddb28f5f9b9dc2c0f9c0b981980310c1053
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | FLOAT32 | batchsize x 3 x 240 x 240 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小     | 数据排布格式 |
  | --------| -------- | -------- | ------------ |
  | output  | FLOAT32  | batchsize x 1000 | ND           |


# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动  

  **表 1**  版本配套表

  | 配套                                                         | 版本    | 环境准备指导                                                 |
  | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
  | 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
<<<<<<< HEAD
  | CANN                                                         | 6.0.RC1 | [CANN推理架构准备](https://www/hiascend.com/software/cann/commercial) |
=======
  | CANN                                                         | 5.1.RC2 | [CANN推理架构准备](https://www/hiascend.com/software/cann/commercial) |
>>>>>>> b50bfddb28f5f9b9dc2c0f9c0b981980310c1053
  | Python                                                       | 3.7.5   | 创建anaconda环境时指定python版本即可，conda create -n ${your_env_name} python==3.7.5 |
  | PyTorch                                                      | 1.11.0   | -                                                            |
  | 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

<<<<<<< HEAD
   ```
=======
   ```sh
>>>>>>> b50bfddb28f5f9b9dc2c0f9c0b981980310c1053
   git clone https://github.com/facebookresearch/pycls
   cd pycls
   git reset f20820e01eef7b9a47b77f13464e3e77c44d5e1f --hard
   cd ..
   ```

2. 安装依赖，测试环境时可能已经安装其中的一些不同版本的库，故手动测试时不推荐使用该命令安装

   ```
   pip3.7 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

    本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的meta.mat。
    
    数据目录结构参考如下格式：

    ```text
    ├──ILSVRC2012_img_val
    ├──val
    ├──ILSVRC2012_devkit_t12
       ├── data
           └── meta.mat
    ```

2. 数据预处理
   1. 首先运行数据集切分脚本ImageNet_val_split.py切分官方val数据集，形成上述目录结构，
      ```
      python3.7 ImageNet_val_split.py ./val ./ILSVRC2012_devkit_t12
      ```
      - 参数说明：

         -   ./val：下载且未分类的ImageNet的val数据集**绝对路径**（如果需要保留val文件夹请先备份）。
         -   ./ILSVRC2012_devkit_t12：官方提供的deckit文件夹**绝对路径**。

   2. 然后将原始数据集转换为模型输入的数据，执行Efficient-B1_preprocess.py脚本，完成预处理。
      ```
      python3.7 Efficient-B1_preprocess.py ./val ./prep_dataset
      ```
      - 参数说明：

         -   ./val：同上述val数据集**绝对路径**。
         -   ./prep_dataset：保存数据集处理后二进制文件的文件夹**绝对路径**。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

<<<<<<< HEAD
      ```
=======
      ```sh
>>>>>>> b50bfddb28f5f9b9dc2c0f9c0b981980310c1053
      wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/EfficientNet-B1/PTH/EN-B1_dds_8gpu.pyth
      ```

   2. 导出onnx文件。

      1. 使用pth2onnx导出onnx文件。

         运行pth2onnx脚本。

         ```
         python3.7 Efficient-B1_pth2onnx.py
         ```

         获得Efficient-b1.onnx文件。

      2. 优化ONNX文件。

<<<<<<< HEAD
         ```
=======
         ```sh
>>>>>>> b50bfddb28f5f9b9dc2c0f9c0b981980310c1053
         python3.7 -m onnxsim --overwrite-input-shape="image:8,3,240,240" ./Efficient-b1.onnx efficient_B1_onnxsim.onnx
         ```

         获得efficient_B1_onnxsim.onnx文件。

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
          atc --model=efficient_B1_onnxsim.onnx --framework=5 --input_format=NCHW --input_shape="image:8,3,240,240" --output=Efficientnet_b1_bs8 --soc_version=Ascend${chip\_name\} --log=debug 
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --input_format：输入数据的格式。
           -   --input_shape：输入数据的shape。
           -   --output：输出的OM模型。
           -   --soc_version：处理器型号。
           -   --log：日志级别。

           运行成功后生成<u>***Efficientnet_b1_bs8.om***</u>模型文件。

2. 开始推理验证。

   1. 使用ais_bench工具进行推理。

<<<<<<< HEAD
      ais_bench工具获取及使用方式请点击查看[ais_bench推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)
=======
      ais_bench工具获取及使用方式请点击查看[ais_bench推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)，文档中aclruntime也需要安装。
>>>>>>> b50bfddb28f5f9b9dc2c0f9c0b981980310c1053

   2. 建立软链接
      将prep_dataset文件夹处理为工具可以输入的格式。
      1. 创建用于保存软链接的文件夹
         ```
         mkdir soft_link
         cd soft_link
         ```
      
      2. 建立软链接（若无法建立，可尝试切换root用户重新建立）
         ```
         find /home/${username}/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/classfication/EfficientNet-B1/prep_dataset/ -name "*.bin" | xargs -i ln -sf {} /home/${username}/ModelZoo-PyTorch/ACL_PyTorch/contrib/cv/classfication/EfficientNet-B1/soft_link/
         ```

   3. 执行推理。

        ```
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        python3.7 -m ais_bench --model Efficientnet_b1_bs8.om --input ./soft_link --output ./ --outfmt TXT --device 0  
        ```

        -   参数说明：

             -   --model：om文件路径。
             -   --input：数据预处理后保存文件的路径。
             -   --output：输出文件夹路径。
             -   --outfmt：输出格式（一般为BIN或者TXT）。
             -   --device：NPU的ID，默认填0。

        推理后的输出默认在当前目录生成{20xx_xx_xx-xx_xx_xx}文件夹。



   4. 精度验证。

      调用Efficient-B1_postprocess.py脚本，可以获得精度accuracy数据（top1和top5），输入指令后请稍等片刻

      ```
       python3.7 Efficient-B1_postprocess.py --pre_dir ${20xx_xx_xx-xx_xx_xx} --data_dir ../val/ --save_file ./result.json
      ```

      - 参数说明：

        -   --pre_dir：为生成推理结果所在相对路径。  
        -   --data_dir：同上述val数据集绝对路径。
        -   --save_file：保存精度验证结果的路径文件。

   5. 性能验证。

      可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

        ```
         python3.7 -m ais_bench --model ./Efficientnet_b1_bs8.om --loop 5
        ```

      - 参数说明：
        - --model：om模型的路径



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size | 数据集|  精度TOP1 | 精度TOP5 | 性能|
| --------- | ----| ----------| ------     |---------|---------|
| 310P3 |  1       | ImageNet |   75.940     |   92.774  |   838.223      |
| 310P3 |  4       | ImageNet |   75.940     |   92.774  |    1235.712      |
| 310P3 |  8       | ImageNet |   75.940     |   92.774  |  1409.692     |
| 310P3 |  16       | ImageNet |   75.940     |   92.774  |   1360.392      |
| 310P3 |  32       | ImageNet |   75.940     |   92.774  |   1304.791      |
| 310P3 |  64       | ImageNet |   75.940     |   92.774  |   1273.800      |