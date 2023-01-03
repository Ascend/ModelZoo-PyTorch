

# SETR模型-推理指导


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

SETR包含一个纯 transformer（即，不进行卷积和分辨率降低）将图像编码为一系列patch。在 transformer的每一层中建模全局上下文，此编码器可以与简单的解码器组合，成为提供功能强大的分割模型。


- 参考实现：

```shell
url=https://github.com/fudan-zvg/SETR 
branch=master
commit_id=23f8fde88182c7965e91c28a0c59d9851af46858
```

适配昇腾 AI 处理器的实现：

```shell
url=https://gitee.com/ascend/ModelZoo-PyTorch
tag=v.0.4.0
code_path=ACL_PyTorch/contrib/cv/segmentation
```


  通过Git获取对应commit\_id的代码方法如下：

  ```shell
  git clone {repository_url}    # 克隆仓库的代码
  cd {repository_name}    # 切换到模型的代码仓目录
  git checkout  {branch}    # 切换到对应分支
  git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id  （可选）
  cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小                      | 数据排布格式 |
  | -------- | -------- | ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 3 x 768 x 768 | NCHW         |


- 输出数据

  | 输出数据 | 大小                       | 数据类型 | 数据排布格式 |
  | -------- | -------------------------- | -------- | ------------ |
  | output1  | batchsize x 19 x 768 x 768 | FLOAT32  | NCHW         |




# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 22.0.2  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.8.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

**表 2**  版本依赖表

| 依赖              | 版本     | 环境准备指导 |
| ----------------- | -------- | ------------ |
| onnx              | 1.9.0    | -            |
| onnx-simplifier   | 0.3.6    | -            |
| Torch             | 1.8.0    | -            |
| TorchVision       | 0.9.0    | -            |
| numpy             | 1.21.2   | -            |
| Pillow            | 8.3.1    | -            |
| opencv-python     | 4.5.3.56 | -            |
| cityscapesScripts | 2.2.0    | -            |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
   git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
   cd /ModelZoo-PyTorch/ACL_PyTorch\contrib\cv\segmentation\SETR
   ```

   文件结构如下：
   
   ```shell
   ├── LICENSE
   ├── SETR_postprocess.py				//数据集后处理评估脚本
   ├── README.md						//说明文档
   ├── requirement.txt					//配置清单
   ├── SETR.patch						//程序补丁
   ├── combine.py						//处理后数据集合并脚本
   └── SETR_preprocess.py				// 数据集前处理脚本
   ```
   
2. 安装依赖。

   ```shell
   pip install -r requirements.txt
   ```

3. 获取开源代码仓。

   （1）手动编译安装1.2.7版本的mmcv。

   ```shell
   git clone https://github.com/open-mmlab/mmcv.git
   cd mmcv
   git checkout v1.2.7
   MMCV_WITH_OPS=True python setup.py build_ext --inplace
   MMCV_WITH_OPS=1 pip install -e .
   cd ..
   ```

   （2）下载源码包。

   ```shell
   git clone https://github.com/fudan-zvg/SETR.git
   cd SETR
   git reset --hard 23f8fde88182c7965e91c28a0c59d9851af46858
   patch -p1 < ../SETR.patch
   pip install -e .
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

   （1）本模型使用CityScape 500张图片的验证集。从[cityscape数据集](https://www.cityscapes-dataset.com/)获取gtFine_trainvaltest.zip和leftImg8bit_trainvaltest.zip，上传数据集到服务器任意目录并解压（如：/root/datasets）。数据集需要软连接到SETR/data文件下。

   （2）数据集目录结构如下：

   ```shell
   cityscapes
   ├── gtFine
   │   ├── test
   │   ├── train
   │   ├── val
   ├── leftImg8bit
   │   ├── tesy
   │   ├── train
   │   ├── val
   ```

2. 数据预处理。

   （1）处理数据集，并将数据集软连接到SETR/data文件下。

   ```shell
   cd SETR   #进入开源代码仓
   mkdir data
   ln -s /root/datasets/cityscapes ./data
   python tools/convert_datasets/cityscapes.py ./data/cityscapes --nproc 8 
   ```
   
   “tools/convert_datasets/cityscapes.py”：数据集预处理脚本路径。
   
   “./data/cityscapes”：数据集路径。
   
   执行cityscapes.py文件时会提示不存在train_extra文件，可以忽略，推理时不会用到。
   
   （2）生成二进制文件。
   
   ```shell
   python ../SETR_preprocess.py configs/SETR/SETR_Naive_768x768_40k_cityscapes_bs_8.py ../input_bin
   cd ..  #退回模型文件夹
   ```
   
   “configs/SETR/SETR_Naive_768x768_40k_cityscapes_bs_8.py”：模型配置文件路径。
   
   “../input_bin”：预处理后的文件保存路径。
   
   运行后生成“input_bin”文件夹包含了预处理后的图片二进制文件。
   
   （3）. 预处理完的数据用combine.py脚本合并到一个文件夹中。
   
   新建new_input_bin文件夹。
   
   ```
   python combine.py ./input_bin
   ```
   
   “./input_bin” 为上一步数据预处理结果目录。合并后数据保存在new_input_bin文件夹中。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   （1）获取权重文件。

   ```
   wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/SETR/PTH/SETR_Naive_cityscapes_b8_40k.pth
   ```
   在SETR文件夹下新建author_pth文件夹，然后把权重文件放入该文件夹

   （2）导出onnx文件。

   使用“pytorch2onnx.py”导出onnx文件。

   ```shell
   python  SETR/tools/pytorch2onnx.py  SETR/configs/SETR/SETR_Naive_768x768_40k_cityscapes_bs_8.py   --checkpoint SETR/author_pth/SETR_Naive_cityscapes_b8_40k.pth   --shape 768 768  --output-file setr_naive_768x768_bs1.onnx
   ```

   “SETR/configs/SETR/SETR_Naive_768x768_40k_cityscapes_bs_8.py”为模型配置文件的路径。

   --checkpoint ：权重文件。

   --shape：输入图形参数大小。

   --output-file：生成的onnx文件。

   使用onnxsim对onnx进行优化。

   ```
   python -m onnxsim setr_naive_768x768_bs1.onnx setr_naive_768x768_sim.onnx  --input-shape=1,3,768,768
   ```
   
   “setr_naive_768x768_bs1.onnx”为输入的文件，“setr_naive_768x768_sim.onnx”为优化后的文件。
   
   --input-shape：输入图形参数大小。
   
   运行后得到“setr_naive_768x768_sim.onnx”文件。
   

   （3）使用ATC工具将ONNX模型转OM模型。

   a.配置环境变量。

   ```shell
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

     > **说明：** 
     >该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

   b.执行命令查看芯片名称（$\{chip\_name\}）。

   ```
      npu-smi info
      #该设备芯片名为Ascend310P3 （自行替换）
      回显如下：
      +--------------------------------------------------------------------------------------------+
      | npu-smi 22.0.0                       Version: 22.0.2                                       |
      +-------------------+-----------------+------------------------------------------------------+
      | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
      | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
      +===================+=================+======================================================+
      | 0       310P3     | OK              | 16.0         50                1236 / 1236           |
      | 0       0         | 0000:86:00.0    | 0            4066 / 21534                            |
      +===================+=================+======================================================+
   ```

   c.执行ATC命令。

      ```shell
      atc --framework=5 --model=setr_naive_768x768_sim.onnx --output=setr_naive_768x768_bs1 --input_format=NCHW --input_shape="img:1,3,768,768" --log=debug --soc_version=Ascend {chip_name}
      ```
      参数说明：
      -   --model：为ONNX模型文件。
      -   --framework：5代表ONNX模型。
      -   --output：输出的OM模型。
      -   --input_format：输入数据的格式。
      -   --input_shape：输入数据的shape。
      -   --log：日志级别。
      -   --soc_version：处理器型号。

      运行成功后生成"setr_naive_768x768_bs1.om"模型文件。

2. 开始推理验证。

   1. a. 使用ais-infer工具进行推理。

      ais-infer工具获取及使用方式请点击查看[[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

      b.  执行推理与性能验证。

      ```shell
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      # 执行推理
      python -m ais_bench --model new_setr_naive_768x768_bs1.om --input ./new_input_bin/ --output ./lcmout/ --outfmt BIN --batchsize 1
      ```
      
      参数说明：

      -   --model：需要进行推理的om模型
      -   --input：模型需要的输入，支持bin文件和目录，若不加该参数，会自动生成都为0的数据。
      -   --output：推理结果输出路径。默认会建立日期+时间的子文件夹保存输出结果。
      -   --outfmt：输出数据的格式，默认”BIN“，可取值“NPY”、“BIN”、“TXT”。
      -   --batchsize：模型batch size。

3. 精度验证。

   ```shell
   python ../SETR_postprocess.py ./configs/SETR/SETR_Naiv68x768_40k_cityscapes_bs_8.py ../lcmout/output/ ../lcmout/merge_output ./miou_eval_result	
   ```

   "../lcmout/output/" 为推理结果存储目录。

   “../lcmout/merge_output” 整合后的推理结果目录。

   “./miou_eval_result” 精度验证结果存储目录。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   | 数据集 | 精度 | 性能 |
| --------- | ---------------- | ---------- | ---------- | --------------- |
| Ascend310P3 | 1 | CityScape | mIoU:77.35 | 3.4376 |