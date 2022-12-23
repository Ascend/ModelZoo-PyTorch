# Segformer 模型-推理指导


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

Segformer是一个简单、高效但功能强大的语义分割框架，它将 Transformer 与轻量级多层感知 (MLP) 解码器相结合。SegFormer 有两个吸引人的特性：1) SegFormer 包含一个新颖的分层结构的 Transformer 编码器，它输出多尺度特征。它不需要位置编码，从而避免了在测试分辨率与训练不同时导致性能下降的位置编码插值。2) SegFormer 避免了复杂的解码器。所提出的 MLP 解码器聚合来自不同层的信息，从而结合局部注意力和全局注意力来呈现强大的表示。


- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segformer
  branch=master
  commit_id=0e37281884193838417a43802bb7a4c854d2067e
  model_name=Segformer-MIT-B0
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
  | input    | RGB_FP32 | batchsize x 3 x 1024 x 2048 | NCHW         |


- 输出数据

  | 输出数据 | 大小     | 数据类型 | 数据排布格式 |
  | -------- | -------- | -------- | ------------ |
  | output  | 1 x batchsize x 1024 x 2048 | INT64  | NCHW    |


# 推理环境准备\[所有版本\]<a name="ZH-CN_TOPIC_0000001126281702"></a>

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

| 配套                                                         | 版本    | 环境准备指导                                                 |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.7.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取源码。

   ```
    git clone https://github.com/open-mmlab/mmsegmentation.git
    cd mmsegmentation
    git reset --hard 0e37281884193838417a43802bb7a4c854d2067e
    patch -p1 < ../Segformer.patch
    pip3 install -v -e .
    cd ..
   ```

2. 安装依赖。

   ```
   # 可参考以下步骤配置环境：

    pip3 install torch==1.7.0 torchvision==0.8.0
    pip3 install mmcv-full==1.4.3 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.7.0/index.html
    pip3 install -r requirements.txt

    git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
    cd MagicONNX
    git checkout dev
    git reset --hard cb071bb62f34bfae405af52063d7a2a4b101358a
    pip3 install .
    cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。

   下载 [cityscpaes数据集](https://www.cityscapes-dataset.com)，解压后数据集目录结构如下：
   ```
    |-- cityscapes
        |-- gtFine
            |-- test
            |-- train
            |-- val
        |-- leftImg8bit
            |-- test
            |-- train
            |-- val
   ```

2. 数据预处理。

   将原始数据集转换为模型输入的二进制数据。

   执行Segformer_preprocess.py脚本，完成预处理。

   ```
   python3 ./Segformer_preprocess.py ${data_path}/cityscapes/leftImg8bit/val ./prep_dataset
   ```
   ${data_path}/cityscapes/leftImg8bit/val：验证集路径。
    ./prep_dataset：预处理后的 bin 文件存放路径。


## 模型推理<a name="section741711594517"></a>

1. 模型转换。

   使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

   1. 获取权重文件。

      wget https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth

   2. 导出onnx文件。

      1. 使用pytorch2onnx.py导出onnx文件。

         运行pytorch2onnx.py脚本。

         ```
         python3 mmsegmentation/tools/pytorch2onnx.py \
         mmsegmentation/configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py \
         --checkpoint segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth \
         --output-file segformer_dynamicbs.onnx \
         --shape 1024 2048 \
         --dynamic-export
         ```

         获得segformer_dynamicbs.onnx文件。

      2. 优化ONNX文件。

         使用onnx-simplifier简化onnx模型
         ```
         onnxsim segformer_dynamicbs.onnx segformer_dynamicbs_sim.onnx
         ```

         使用optimize_onnx.py优化onnx模型
         ```
         python3 optimize_onnx.py segformer_dynamicbs_sim.onnx segformer_dynamicbs_sim_opt.onnx
         ```

         获得segformer_dynamicbs_sim_opt.onnx文件。

   3. 使用ATC工具将ONNX模型转OM模型。

      1. 配置环境变量。

         ```
         source /usr/local/Ascend/ascend-toolkit/set_env.sh
         ```

         > **说明：** 
         > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      2. 执行命令查看芯片名称（$\{chip\_name\}）。

         ```
         npu-smi info
         # 该设备芯片名为 Ascend310P3
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
         atc --framework=5 \
         --model=segformer_dynamicbs_sim_opt.onnx \
         --output=segformer_bs${batch_size} \
         --input_format=NCHW \
         --input_shape="input:${batch_size},3,1024,2048" \
         --soc_version=Ascend${chip_name} \
         --log=error
         ```

         - 参数说明：

           -   --model：为ONNX模型文件。
           -   --framework：5代表ONNX模型。
           -   --output：输出的OM模型。
           -   --input\_format：输入数据的格式。
           -   --input\_shape：输入数据的shape，${batch_size}的值可取：1，4。
           -   --log：日志级别。
           -   --soc\_version：处理器型号。
           
           运行成功后生成 segformer_bs${batch_size}.om 模型文件。



2. 开始推理验证。

   a.  安装ais_bench推理工具。

      请点击本链接进行安装ais_bench推理工具，以及查看具体使用方法(https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)  


   b.  执行推理。

      ```
      python3 -m ais_bench \
      --model=./segformer_bs${batch_size}.om \
      --input=./prep_dataset \
      --output=./ \
      --outfmt=BIN
      ```

      -   参数说明：

           -   --model：om文件路径。
           -   --input：预处理后的数据集路径。
           -   --output：输出文件保存路径。
           -   --outfmt：输出文件格式。

      `${batch_size}` 表示不同 batch 的 om 模型，该模型支持的batchsize为：1，4。

      推理后的输出默认在当前工作目录下，其目录命名格式为`xxxx_xx_xx-xx_xx_xx`(`年_月_日-时_分_秒`)，如`2022_08_30-08_50_53`。

      > **说明：** 
      > 执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见[《ais_bench 推理工具使用文档》](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)。

   c.  精度验证。

      我们将后处理与精度验证一起放在了Segformer_postprocess.py文件中，脚本执行完毕后即可生成精度结果。

      ```
      python3 Segformer_postprocess.py --json_path=${path_to_json}/sumary.json --dataset_path=${data_path}
      ```

      -   参数说明：

           -   --json_path：ais_bench工具生成的json文件路径；${path_to_json}代表sumary.json文件的存放路径。
           -   --dataset_path：cityscpaes数据集所在路径；比如：若cityscpaes存放在/opt/npu/cityscpaes，则--dataset_path=/opt/npu/


   d.  性能验证。

      使用ais_bench推理工具的纯推理模式验证不同batch_size的om模型的性能，命令如下：

      ```
      python3 -m ais_bench --model=./segformer_bs${batch_size}.om --loop=50 --batchsize=${batch_size}
      ```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能参考下列数据。

| 芯片型号 | Batch Size   |   数据集   |    精度    |       性能       |
| --------- | ---------- | ---------- | ---------- | --------------- |
| 310P      |    1       | cityscapes |  mIoU = 75.94  |  6.37 fps  |
| 310P      |    4       | cityscapes |  mIoU = 75.94  |  5.60 fps  |
注：该模型支持的batchsize为1，4。
