# Pix2Pix模型-推理指导

- [概述](##概述)
- [输入输出数据](##输入输出数据)
- [推理环境准备](##推理环境准备)
- [快速上手](##快速上手)
   - [准备数据集](###准备数据集)
- [模型推理](##模型推理)
- [模型推理精度&性能](##模型推理精度&性能)
   - [精度](##精度)
   - [性能](##性能)

## 概述

Pix2pix是一个图像合成网络，是将GAN应用于有监督的图像到图像翻译的经典论文。其是将CGAN的思想运用在了图像翻译的领域上，学习从输入图像到输出图像之间的映射，从而得到指定的输出图像。

- 参考论文：

  P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. Image-toimage translation with conditional adversarial networks. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

- 参考实现：

  ```
  url=https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
  branch=master 
  commit_id=aac572a869b6cfc7486d1d8e2846e5e34e3f0e05
  model_name=Pix2Pix
  ```
  
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/modelzoo.git 
  branch=master 
  commit_id=676b142ab9e3068ca0f4ef77825b1f55454b6e09
  code_path=/contrib/ACL_PyTorch/Research/cv/gan/Pix2Pix
  ```
  
  通过Git获取对应commit_id的代码方法如下：

  ```
  git clone {repository_url}        # 克隆仓库的代码 
  cd {repository_name}              # 切换到模型的代码仓目录 
  git checkout {branch}             # 切换到对应分支 
  git reset --hard {commit_id}      # 代码设置到对应的commit_id 
  cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

## 输入输出数据

- 输入数据

   | 输入数据 | 大小                      | 数据类型 | 数据排布格式 |
   | -------- | ------------------------- | -------- | ------------ |
   | inputs   | batchsize x 3 x 256 x 256 | RGB_FP32 | NCHW         |

- 输出数据

   | 输出数据  | 大小                     | 数据类型 | 数据排布格式 |
   | --------- | ------------------------ | -------- | ------------ |
   | NetOutput | batchsize x3 x 256 x 256 | RGB_FP32 | NCHW         |

## 推理环境准备

- 本样例配套的CANN版本为CANN 5.1.RC2  。

- 硬件环境、开发环境和运行环境准备请参见《[CANN 软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-upgrade)》。

- 该模型需要以下依赖。请确认安装依赖。

  **表 1** 版本配套表

| 依赖名称    | 版本                        | 默认版本 |
| ----------- | --------------------------- | -------- |
| CANN        | 5.1.RC2                     | -        |
| onnx        | 1.7.0                       | 1.7.0    |
| torch       | 1.5.0+ascend.post5.20220315 | None     |
| torchvision | 0.6.0                       | None     |
| numpy       | 1.21.0                      | None     |
| Pillow      | 7.2.0                       | None     |
| decorator   | 5.0.9                       | None     |
| sympy       | 1.9                         | None     |
| dominate    | 2.7.0                       | None     |
| aclruntime  | 0.0.1                       | None     |
| tqdm        | 4.64.0                      | None     |

## 快速上手

### 准备数据集

1. 获取原始数据集

   以facades为例，请用户需自行获取facades数据集，上传数据集到Pix2Pix的datasets目录下并解压（如：/Pix2Pix/datasets/facades）。训练集和验证集图片分别位于“train/”和“test/”文件夹路径下。  
   
   [数据集链接](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/)

   ```
   ├Pix2Pix
   ├─datasets
   ├── facades
   │  ├──train├──图片1、2、3、4
   │  │        
   │  ├──test ├──图片1、2、3、4
   ```

2. 数据预处理

   数据预处理将原始数据集jpg转换为模型输入的数据bin。

- BIN 文件输入

   ```bash
   python pix2pix_preprocess.py --dataroot './datasets/facades' 
   ```

## 模型推理

1. 模型转换

   1. 获取权重文件（两种方法）

      1. 运行*download_pix2pix_model.sh*脚本下载权重文件
      2. 下载Pix2Pix中facades_label2photo_pretrained的权重文件latest_net_G.pth，放到./checkpoints/facades_label2photo_pretrained目录下。

   2. 导出onnx文件

      1. 使用latest_net_G.pth导出onnx文件

         运行pth2onnx.py脚本。

         ```bash
         python pix2pix_pth2onnx.py \
         --direction BtoA \ # 无需修改  
         --model pix2pix \ # 无需修改 
         --checkpoints_dir ./checkpoints \ # 路径
         --name facades_label2photo_pretrained  # 文件名
         ```
         获得netG_onnx.onnx文件。
         
   3. 使用ATC工具将ONNX模型转为OM模型

       1. 配置环境变量
       
          ```bash
          source /usr/local/Ascend/ascend-toolkit/set_env.sh
          ```
       
          > **说明：** 该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 (推理)](https://gitee.com/link?target=https%3A%2F%2Fsupport.huawei.com%2Fenterprise%2Fzh%2Fascend-computing%2Fcann-pid-251168373%3Fcategory%3Ddeveloper-documents%26subcategory%3Dauxiliary-development-tools)》。
       
       2. 执行命令查看芯片名称($\{chip\_name\})
       
          ```bash
          npu-smi info
          #该设备芯片名为Ascend310P3(自行替换)
          结果如下：
          +--------------------------------------------------------------------------------------------+
          | npu-smi 22.0.0                       Version: 22.0.2                                       |
          +-------------------+-----------------+------------------------------------------------------+
          | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
          | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
          +===================+=================+======================================================+
          | 0       310P3     | OK              | 16.5         56                0    / 0              |
          | 0       0         | 0000:3B:00.0    | 0            924  / 21534                            |
          +===================+=================+======================================================+
          ```
       
       3. 执行ATC命令
       
          ```bash
          atc --framework=5 --model=./checkpoints/facades_label2photo_pretrained/netG_onnx.onnx --output=./checkpoints/facades_label2photo_pretrained/netG_om_bs1 --input_format=NCHW --input_shape="inputs:1,3,256,256" --log=debug --soc_version=Ascend{chip_name}
          ```
       
          - 参数说明：
       
            - --model：为ONNX模型文件。
            - --framework：5代表ONNX模型。
            - --output：输出的OM模型。
            - --input_format：输入数据的格式。
            - --input_shape：输入数据的shape。
            - --log：日志级别。
            - --soc_version：处理器型号。{chip_name}请自行替换为上述的芯片名称。
       
            运行成功后生成netG_om_bs1.om模型文件。

2. 开始推理验证

   1. 使用ais-infer工具进行推理。

      1. 安装推理工具

         ```bash
         git clone https://gitee.com/ascend/tools.git
         cd tools/ais-bench_workload/tool/ais_infer/backend/
         pip3 wheel ./
         pip3 install ./aclruntime-0.0.1-cp37-cp37m-linux_aarch64.whl
         ```

      2. 创建结果输出目录

         ```bash
         mkdir ./result
         ```

      3. 执行推理
   
         ```bash
         python ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model ./checkpoints/facades_label2photo_pretrained/netG_om_bs1.om --input  "./datasets/facades/bin" --output "result/dumpOutput_device0_bs1" --outfmt BIN --batchsize 1
         ```

         - 参数说明：
         - --model: 需要进行推理的om模型。
         - --input：模型需要的输入，支持bin文件和目录。
         -  --output：推理结果输出路径。
         - --outfmt：输出数据的格式。

         推理后的输出在目录./result/dumpOutput_device0_bs1/Timestam下，Timestam为日期+时间的子文件夹,如 2022_08_11-10_37_29。为方便执行后续精度验证，可以更改输出文件夹的名字。
         
         ```bash
         cd ./result/dumpOutput_device0_bs1
         mv 2022_08_11-10_37_29/ bs1
         ```
   
   
   2. 数据后处理
   
      将推理的bin文件转为jpg
   
      ```bash
      python3.7 pix2pix_postprocess.py --bin2img_file=./result/bin2img_bs1/  --npu_bin_file=./result/dumpOutput_device0_bs1/bs1
      ```
   
      - 参数说明
        - --bin2img_file 为图片保存路径
        - --npu_bin_file 为推理输出bin文件路径
   
      转化的jpg文件在./result/bin2img_bs1/下参看
   
   3. 精度验证。
   
      根据源码仓根据作者pth生成图片命令
   
      ```bash
      python test.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained
      ```
   
      - 转化的jpg文件在./results/facades_label2photo_pretrained/test_latest/images查看
   
      - 将pth生成的图片与om生成的图片做比较

## 模型推理精度&性能

### 性能
| Model     | Batch Size | 310 (FPS/Card) | 310P (FPS/Card) | T4 (FPS/Card) | 310P/310   | 310P/T4      |
| --------- | ---------- | -------------- | -------------- | ------------- | --------- | ----------- |
| Pix2Pix   | 1          | 548.63952      | 527.00922      | 509.4840455   | 0.9605747 | 1.034397894 |
| Pix2Pix   | 4          | 500.15276      | 872.76959      | 409.9146045   | 1.745006  | 2.129149769 |
| Pix2Pix   | 8          | 497.8626       | 743.1343       | 344.3259389   | 1.4926494 | 2.158229227 |
| Pix2Pix   | 16         | 503.57928      | 920.12192      | 335.3932066   | 1.827164  | 2.743412512 |
| Pix2Pix   | 32         | 500.53896      | 924.53483      | 295.1675537   | 1.8470787 | 3.132237332 |
| Pix2Pix   | 64         | 498.2512       | 915.08315      | 282.4260397   | 1.83659   | 3.24008068  |
| 最优batch |            | 548.63952      | 924.53483      | 509.4840455   | 1.6851408 | 1.81464923  |

### 精度

![原图](https://foruda.gitee.com/images/1660789893035211565/2_real_b.png "2_real_B.png") | ![窗口](https://foruda.gitee.com/images/1660789909317587071/2_real_a.png "2_real_A.png")
---|---
原图 | 窗口

![pth处理结果](https://foruda.gitee.com/images/1660789926632865129/2_fake_b.png "2_fake_B.png") | ![om模型batchsize=1处理结果](https://foruda.gitee.com/images/1660789949522235347/2_0.jpeg "2_0.jpg") | ![om模型batchsize=16处理结果](https://foruda.gitee.com/images/1660789966155300818/2_0.jpeg "2_0.jpg")
---|---|---
pth处理结果 | om模型batchsize=1处理结果 | om模型batchsize=16处理结果



