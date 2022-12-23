# EfficientNet-B3模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

- [推理环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型推理](#section741711594517)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

- [配套环境](#ZH-CN_TOPIC_0000001126121892)

  ******



# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

EfficientNet-B3是一种卷积神经网络，该网络是在EfficientNet-B0的基础上，利用NAS搜索技术，对输入分辨率Resolution、网络深度Layers、网络宽度Channels三者进行综合调整的结果。

- 参考实现：

  ```
  url=https://github.com/facebookresearch/pycls.git
  commit_id=f20820e01eef7b9a47b77f13464e3e77c44d5e1f
  model_name=EfficientNet-B3
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
  | input    | FLOAT32 | batchsize x 3 x 300 x 300 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output1  | 1 x 1000 | FLOAT32  | ND           |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本      | 环境准备指导                                                 |
| ------------------------------------------------------------ |---------| ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.6.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://github.com/facebookresearch/pycls
   cd pycls  
   git reset f20820e01eef7b9a47b77f13464e3e77c44d5e1f --hard  
   cd ..
   ```

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）
   
    该模型使用[ImageNet官网](http://www.image-net.org)的5万张验证集进行测试，以ILSVRC2012为例，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到ILSVRC2012_img_val.tar验证集及ILSVRC2012_devkit_t12.gz中的val_label.txt数据标签。
    
       ├── ImageNet
         ├── ILSVRC2012_img_val
         ├── val_label.txt

2. 数据预处理。

   数据预处理将原始数据集（.jpeg）转换为模型输入的二进制文件（.bin）。

   执行imagenet_torch_preprocess.py脚本，完成预处理。

    ```
   python3.7 imagenet_torch_preprocess.py efficientnetB3 ./dataset/ImageNet/ILSVRC2012_img_val ./prep_dataset
    ```

     - 参数说明：
       - efficientnetB3：为模型名称。
       - ./dataset/ImageNet/ILSVRC2012_img_val：为验证集路径。
       - ./prep_dataset：为预处理后生成的二进制文件的存储路径。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。
   
      [EfficientNet-B3预训练pth权重文件](https://dl.fbaipublicfiles.com/pycls/dds_baselines/161305060/EN-B3_dds_8gpu.pyth)  
      文件md5sum: 4c809d9cb292ce541f278d11899e7b38 
         ```
         wget https://dl.fbaipublicfiles.com/pycls/dds_baselines/161305060/EN-B3_dds_8gpu.pyth
         ```

   2. 导出onnx文件。

      1. 使用efficientnetB3_pth2onnx.py导出onnx文件。

         运行efficientnetB3_pth2onnx.py脚本。

         ```
         python3.7 efficientnetB3_pth2onnx.py EN-B3_dds_8gpu.pyth ./pycls/configs/dds_baselines/effnet/EN-B3_dds_8gpu.yaml efficientnetB3.onnx
         ```

         获得efficientnetB3.onnx文件。

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
         ```
         atc --framework=5 --model=./efficientnetB3.onnx --input_format=NCHW --input_shape="image:1,3,300,300" --output=efficientnetB3_bs1 --log=debug --soc_version=Ascend${chip_name}  
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           -   --insert\_op\_conf=aipp\_resnet34.config:  AIPP插入节点，通过config文件配置算子信息，功能包括图片色域转换、裁剪、归一化，主要用于处理原图输入数据，常与DVPP配合使用，详见下文数据预处理。

           运行成功后生成efficientnetB3_bs1.om模型文件。



2. 开始推理验证。

   a.  使用ais_bench工具进行推理。

      ais_bench工具获取及使用方式请点击查看[[ais_bench 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]


   b.  执行推理。

   ```
   python3.7 -m ais_bench  --model ./efficientnetB3_bs1.om --input ./prep_dataset  --output ./result_bs1 --outfmt TXT   
   ```

- 参数说明：

  - --model：om文件路径。
  - --input：模型需要的输入(预处理后的生成文件)，支持bin文件和目录，若不加该参数，会自动生成都为0的数据。
  - --output：为推理数据输出路径。
  - --outfmt：输出数据的格式，可取值“NPY”、“BIN”“TXT”，精度验证需要TXT格式。
  - -- loop：为推理次数，可选参数，默认1，profiler为true时，推荐为1。
  

   推理后的输出在output参数对应路径的文件result_bs1里,推理结果保存在sumary.json里，便于汇总统计。

 >**说明：** 
>执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见《ais_bench 推理工具使用文档》。

  c.  精度验证。

调用脚本与数据集标签val\_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

   ```
cd result_bs1
rm -r sumary.json
cd ..
python3.7 imagenet_acc_eval.py ./result_bs1 ./datasets/imagenet/val_label.txt ./ result.json
   ```
 - 参数说明
   - ./result_bs1：为生成推理结果所在路径 
   - val_label.txt：为标签数据 
   - result.json：为生成结果文件


   d.  性能验证。

可使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，参考命令如下：

```
python3.7 -m ais_bench --model=${om_model_path} --loop=20 --batchsize=${batch_size}
```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| Precision | Top1   | Top5   |
|-----------|--------|--------|
| 310精度     | 77.56% | 93.48% |
| 310P精度    | 77.56% | 93.47% |


| Throughput | 310     | 310P     | T4      | 310P/310 | 310P/T4  |
|------------|---------|----------|---------|----------|----------|
| bs1        | 428.316 | 443.341  | 417.77  | 1.035079 | 1.0612   |
| bs4        | 558.568 | 655.738  | 591.656 | 1.17396  | 1.1083   |
| bs8        | 581.524 | 700.918  | 631.597 | 1.2053   | 1.109755 |
| bs16       | 584.288 | 755.444  | 656.825 | 1.29293  | 1.150145 |
| bs32       | 583.068 | 733.2756 | 690.188 | 1.2576   | 1.0624   |
| bs64       | 589.856 | 735.59   | 713.43  | 1.247    | 1.031    |
| 最优batch    | 589.856 | 755.444  | 713.43  | 1.2807   | 1.05889  |