# vit_base_patch32_224模型-推理指导


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

Transformer架构已广泛应用于自然语言处理领域。本模型的作者发现，Vision Transformer（ViT）模型在计算机视觉领域中对CNN的依赖不是必需的，直接将其应用于图像块序列来进行图像分类时，也能得到和目前卷积网络相媲美的准确率。

- 参考实现：

  ```
  url=https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
  commit_id=20b2d4b69dae2ec185a77a50cf1d38d55d94b657
  model_name=vision_transformer
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
| 固件与驱动                                                   | 1.0.15  | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies) |
| CANN                                                         | 5.1.RC2 | -                                                            |
| Python                                                       | 3.7.5   | -                                                            |
| PyTorch                                                      | 1.6.0   | -                                                            |
| 说明：Atlas 300I Duo 推理卡请以CANN版本选择实际固件与驱动版本。 | \       | \                                                            |

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

1. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf  \*.tar与 unzip \*.zip）

    本模型支持ImageNet 50000张图片的验证集。以ILSVRC2012为例，请用户需自行获取ILSVRC2012数据集，上传数据集到服务器任意目录并解压（如：/home/HwHiAiUser/dataset）。本模型将使用到“ILSVRC2012_img_val.tar”验证集及“ILSVRC2012_devkit_t12.gz”中的“ILSVRC2012_validation_ground_truth.txt”数据标签和meta.mat文件。将“ILSVRC2012_validation_ground_truth.txt”和“meta.mat”放到推理目录。

    数据目录结构请参考：

    ```bash
    ├── ImageNet
        ├── ILSVRC2012_img_val
    ```

2. 数据预处理。

   数据预处理将原始数据集转换为模型输入的数据。

   执行“vit_base_patch32_224_preprocess.py”脚本，完成预处理。

   ```
   python3.7 vit_base_patch32_224_preprocess.py --data-path ${datasets_path} --store-path ./prep_dataset
   ```

    --data-path：原始数据验证集（.jpeg）所在路径。

    --store-path：输出的二进制文件（.bin）所在路径。

    每个图像对应生成一个二进制文件。运行成功后，在当前目录下生成“prep_dataset”二进制文件夹。

## 模型推理<a name="section741711594517"></a>

1. 模型转换。

    使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

    1. 获取权重文件。

        将权重文件放到当前工作目录，可以通过以下命令下载，解压后获取权重文件：“B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz”：

        ```
        wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/22.1.30/ATC%20vit_base_patch32_224%28FP16%29%20from%20Pytorch%20-%20Ascend310.zip
        ```

    2. 导出onnx文件。

        使用“B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz”导出onnx文件。

        运行“pth2onnx.py”脚本：

        ```
        python3.7 vit_base_patch32_224_pth2onnx.py --batch-size 1 --model-path B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz
        ```

        获得“vit_bs1.onnx”文件。

        优化ONNX文件：

        ```
        python3.7 -m onnxsim ./vit_bs1.onnx ./vit_bs1_sim.onnx --input-shape "input:1,3,224,224"
        ```

        获得“vit_bs1_sim.onnx”文件。

        > **须知：**
        > 使用ATC工具将.onnx文件转换为.om文件，导出.onnx模型文件时需设置算子版本为11。

   3. 使用ATC工具将ONNX模型转OM模型。

      配置环境变量：

        ```
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        ```

        > **说明：**
        > 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

      执行ATC命令：

        ```
        atc --framework=5 --model=vit_bs1_sim.onnx --output=vit_bs1 --input_format=NCHW --input_shape="input:1,3,224,224" --log=debug --soc_version=${chip_name} --precision_mode=allow_mix_precision --modify_mixlist=ops_info.json
        ```

        ${chip_name}可通过 npu-smi info指令查看。

        - 参数说明：

        -   --model：为ONNX模型文件。
        -   --framework：5代表ONNX模型。
        -   --output：输出的OM模型。
        -   --input\_format：输入数据的格式。
        -   --input\_shape：输入数据的shape。
        -   --log：日志级别。
        -   --soc\_version：处理器型号。

        运行成功后生成<u>***vit_bs1.om***</u>模型文件。

2. 开始推理验证。

- 安装ais_bench推理工具。

    请访问[ais_bench推理工具](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)代码仓，根据readme文档进行工具安装。  

- 执行推理。

    回到推理目录，创建result文件夹。

    ```
    mkdir result
    ```

    执行推理

    ```
    python3 -m ais_bench --model vit_bs1.om --input "./prep_dataset" --batchsize 1 --output result/ --outfmt TXT
    ```

    -   参数说明：

        -   --model：om模型路径。
        -   --input：预处理后的输入数据。
        -   --batchsize：om模型的batchsize。
        -   --output：推理结果存放目录。

        推理后的输出在目录“./result/Timestam”下，Timestam为日期+时间的子文件夹,如 2022_08_11-10_37_29。

    >**说明：** 
    >执行ais_bench工具请选择与运行环境架构相同的命令。参数详情请参见--help命令。

3.  精度验证。

    调用“vit_base_patch32_224_postprocess.py”脚本，数据结果保存在“result.json”中。

    ```
    python3.7 vit_base_patch32_224_postprocess.py --output result_bs1.json --input-dir ./result/2022_08_11-10_37_29/
    ```

    -   参数说明：

        -   --output：精度结果文件。
        -   --input-dir：推理结果所在路径。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能和精度参考下列数据。

1.精度对比
| 模型           | 仓库pth精度 | 310离线推理精度 | 310P离线推理精度 |
| -------------- | ----------- | --------------- | --------------- |
| vit-small bs1  | top1:80.724 | top1:80.714  |  top1:80.726    |
| vit-small bs8 | top1:80.724 | top1:80.714  |  top1:80.726   |

2.性能对比
| Throughput | 310     | 310P    | T4     | 310P/310    | 310P/T4     |
| ---------- | ------- | ------- | ------ | ----------- | ----------- |
| bs1        | 321.2008 | 336.1748 | 353.2507 | 1.05 | 0.95 |
| bs4        | 444.8147 | 492.1917 | 603.9075 | 1.11 | 0.82 |
| bs8        | 433.9089 | 837.5699 | 1021.6487 | 1.93 | 0.82 |
| bs16       | 424.4384 | 823.2256 | 1013.8774 | 1.94 | 0.81 |
| bs32       | 413.3992  | 814.8914  | 1216.1011 | 1.97 | 0.67 |
| bs64       | 400.4219 | 785.9891 | 1177.6265 | 1.96 | 0.66 |
|            |         |         |        |             |             |
| 最优batch  | 444.8147 | 837.5699 | 1216.1011 | 1.88 | 0.69 |