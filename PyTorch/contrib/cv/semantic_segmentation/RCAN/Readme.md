## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** Image Super-Resolution

**版本（Version）：1.0**

**修改时间（Modified） ：2021.08.31**

**大小（Size）：21M**

**框架（Framework）：Pytorch 1.5**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于Pytorch框架的RCAN图像超分辨率网络训练代码**

## 概述

卷积神经网络（CNN）深度对于图像超分辨率（SR）至关重要。然而，更深层的图像SR网络更难训练。低分辨率的输入和特征包含丰富的低频信息，这些信息在通道间被平等对待，因此阻碍了CNN的表征能力。为了解决这些问题，提出了超深剩余通道注意网络（RCAN）。具体地说，RCAN是一种残差中残差（RIR）结构来形成非常深的网络，它由几个具有长跳跃连接的残差组组成。每个剩余组包含一些具有短跳过连接的剩余块。同时，RIR允许通过多个跳转连接绕过丰富的低频信息，使主网络专注于学习高频信息。此外，提出了一种通道注意机制，通过考虑通道之间的相互依赖性，自适应地重新缩放通道特征。大量实验表明，RCAN相对于最先进的方法实现了更好的准确性和视觉改善。

- 参考论文：

  [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758)

- 参考实现：

  [RCAN](https://github.com/yjn870/RCAN-pytorch)

- 适配昇腾 AI 处理器的实现：

  https://gitee.com/ascend/modelzoo/tree/master/contrib/PyTorch/Research/cv/semantic_segmentation/RCAN

- 通过Git获取对应commit_id的代码方法如下：

  commit_id = 1e0b98b7

  ```
  git clone {repository_url}    # 克隆仓库的代码
  cd {repository_name}    # 切换到模型的代码仓目录
  git checkout  {branch}    # 切换到对应分支
  git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
  cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

## 默认配置

- 训练数据集预处理（以DIV2K为例，仅作为用户参考示例）：
  - 图像裁剪成96X96的patch
  - 图像输入格式：png
- 测试数据集预处理（以DIV2K为例，仅作为用户参考示例）
  - 图像的输入尺寸为96X96的patch（将图像最小边缩放到所需要超分辨率的倍数作为低分辨率图像进行输入）
  - 图像输入格式：png
  - 归一化：ToTensor
- 训练超参
  - batch_size：160（1P）1280（8P）
  - patch_size：48
  - lr：1e-4
  - workers：8
  - scale：2
  - num_features：64
  - num_rg：10
  - num_rcab：20
  - reduction：16
  - Train epoch: 600

## 支持特性

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 并行数据   | 是       |

## 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度

脚本已默认开启混合精度，设置训练脚本参考如下。

```
--amp 开启混合精度
```

## 训练环境准备

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南](https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

2. 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

   当前模型支持的镜像列表如[表1](https://gitee.com/ascend/modelzoo/blob/master/built-in/TensorFlow/Official/cv/image_classification/DenseNet121_for_TensorFlow/README.md#zh-cn_topic_0000001074498056_table1519011227314)所示。

   **表 1** 镜像列表

   | *镜像名称*                                                   | *镜像版本* | *配套CANN版本*                                               |
   | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
   | *ARM架构：[ascend-tensorflow-arm](https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm)**x86架构：[ascend-tensorflow-x86](https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86)* | *20.2.0*   | *[20.2](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)* |

## 数据集准备

1、模型训练使用DIV2K数据集，数据集请用户自行获取。

> 请自行在 DIV2K dataset 官网上下载 DIV2K_train_HR.zip和DIV2K_valid_HR.zip
>
> Download the original dataset of DIV2K_HR from the official website. The first one of zip file has 800 png files, and the other one has 100 png.

2、数据集训练前需要做预处理操作，裁剪保存，数据集处理后，放入目录下，在训练脚本中指定数据集路径，可正常使用。

```shell
python3.7 ./dataset_make.py --input_zip_path path1 --dataset_path path2
```

`path1` is your floader path where there are the two zip files you downloaded.

`path2` is your path want to store the final augmented dataset. 

## 模型训练与测试

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

  [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend 910训练平台环境变量设置?sort_id=3148819)

- 1P训练

  进入`test`目录后，运行脚本`train_full_1p.sh`中，配置所需变量

  + 训练数据文件夹路径`real_traindata_path`
  + 测试数据文件夹路径`real_testdata_path`
  + 输出文件夹的路径`real_output_path`

  ```shell
  ./train_full_1p.sh --train_dataset_dir=real_traindata_path --test_dataset_dir=real_testdata_path --outputs_dir=real_output_path 
  # 选择设备 将如下设备变量加入命令（默认选择npu，如果需要使用gpu的话需要使用参数重新制定，npu可以不加）
  --device=npu
  --device=gpu
  ```

+ 1P性能

  进入`test`目录后，运行脚本`train_performance_1p.sh`中，配置所需变量

  + 训练数据文件夹路径`real_traindata_path`
  + 测试数据文件夹路径`real_testdata_path`
  + 输出文件夹的路径`real_output_path`

    ```shell
    ./train_performance_1p.sh --train_dataset_dir=real_traindata_path --test_dataset_dir=real_testdata_path --outputs_dir=real_output_path 
    # 选择设备 将如下设备变量加入命令（默认选择npu，如果需要使用gpu的话需要使用参数重新制定，npu可以不加）
    --device=npu
    --device=gpu
    ```

+ 8P训练

  进入`test`目录后，运行脚本`train_full_1p.sh`中，配置所需变量

  + 训练数据文件夹路径`real_traindata_path`
  + 测试数据文件夹路径`real_testdata_path`
  + 输出文件夹的路径`real_output_path`

  ```shell
  ./train_full_8p.sh --train_dataset_dir=real_traindata_path --test_dataset_dir=real_testdata_path --outputs_dir=real_output_path 
  # 选择设备 将如下设备变量加入命令
  --device=npu
  --device=gpu
  ```

+ 8P性能

  进入`test`目录后，运行脚本`train_performance_8p.sh`中，配置所需变量

  + 训练数据文件夹路径`real_traindata_path`
  + 测试数据文件夹路径`real_testdata_path`
  + 输出文件夹的路径`real_output_path`

    ```shell
    ./train_performance_1p.sh --train_dataset_dir=real_traindata_path --test_dataset_dir=real_testdata_path --outputs_dir=real_output_path 
    # 选择设备 将如下设备变量加入命令
    --device=npu
    --device=gpu
    ```

+ 断点继续训练

  在训练指令`train_full_Xp.sh`中对如下参数进行修改

  ```shell
  --ifcontinue 
  # 举例如下
  ./train_full_8p.sh --train_dataset_dir=real_traindata_path --test_dataset_dir=real_testdata_path --outputs_dir=real_output_path --ifcontinue 
  ```

+ 验证

  进入`test`目录后，运行脚本`run_test.sh`中，配置所需变量

  + 需要测试checkpoint的路径`real_checkpoint_path`
  + 测试数据文件夹路径`real_testdata_path`
  + 输出文件夹的路径`real_output_path`
  
  ```shell
  # 使用gpu测试
  ./run_test.sh --test_dataset_dir=real_testdata_path --outputs_dir=real_output_path --checkpoint_path=real_checkpoint_path --device=gpu
  # 使用gpu测试，如果模型训练保存自多卡任务
  ./run_test.sh --test_dataset_dir=real_testdata_path --outputs_dir=real_output_path --checkpoint_path=real_checkpoint_path --device=gpu --from_multiprocessing_distributed
  # 使用npu测试
  ./run_test.sh --test_dataset_dir=real_testdata_path --outputs_dir=real_output_path --checkpoint_path=real_checkpoint_path --device=npu
  # 使用npu测试，如果模型训练保存自多卡任务
  ./run_test.sh --test_dataset_dir=real_testdata_path --outputs_dir=real_output_path --checkpoint_path=real_checkpoint_path --device=npu --from_multiprocessing_distributed
  ```

+ 微调
  进入`test`目录后，运行脚本`train_finetuning_1p.sh`中，配置所需变量

  + 需要加载的checkpoint的路径`real_checkpoint_path`
  + 训练数据文件夹路径`real_traindata_path`
  + 测试数据文件夹路径`real_testdata_path`
  + 输出文件夹的路径`real_output_path`

  ```shell
  ./train_finetuning_1p.sh --train_dataset_dir=real_traindata_path --test_dataset_dir=real_testdata_path --outputs_dir=real_output_path --finetuning_checkpoint_path=real_checkpoint_path
  ```

## 脚本和示例代码功能

```
├── main.py                                     //网络训练代码
├── main_prof.py                                //网络训练输出prof文件
├── test.py                                     //网络测试代码
├── model.py                                    //网络模型
├── dataset.py                                  //数据集加载
├── datasetmake.py                              //数据集制作
├── utils.py                                    //常用函数
├── README.md                                   //代码说明文档
├── test/
│    ├──train_full_1p.sh                        //单卡完整训练运行启动脚本
│    ├──train_full_8p.sh                        //多卡完整训练运行启动脚本
│    ├──train_performance_1p.sh                 //单卡性能训练运行启动脚本
│    ├──train_performance_8p.sh                 //多卡性能训练运行启动脚本
│    ├──train_finetuning_1p.sh                  //单卡微调训练运行启动脚本
│    ├──run_test.sh                             //测试运行启动脚本
|    ├──run_1p_prof.sh                          //生成prof文件
```

## 脚本参数

```
--arch                          网络模型选择，默认："RCAN"
--train_dataset_dir             训练数据文件路径
--test_dataset_dir              测试数据文件路径
--outputs_dir                   输出文件文件夹
--patch_size                    输入尺寸，默认：48
--batch_size                    批大小，默认：160
--num_epochs                    训练轮数，默认：600
--lr                            学习率，默认：1e-4
--workers                       加载数据线程数，默认：8
--seed                          随机种子，默认：123
--scale                         超分辨率倍数，默认：2
--num_features                  网络模型参数1，默认：64
--num_rg                        网络模型参数2，默认：10
--num_rcab                      网络模型参数3，默认：20
--reduction                     网络模型参数4，默认：16
--ifcontinue                    是否断点继续训练，默认：False
--checkpoint_path               断点训练的存储路径
--iffinetuning                  是否微调训练，默认：False
--finetuning_checkpoint_path    微调训练的存储路径
--amp                           是否混合精度，默认：False
--loss_scale                    混合精度等级，默认：128.0
--opt_level                     混合精度等级，默认：'O2'             
--device                        设备，默认："gpu"
--device_list                   设备列表，默认：'0,1,2,3,4,5,6,7'
--device_id                     单卡选用设备，默认：None
--world_size                    多卡环境数量，默认：1
--multiprocessing_distributed   是否使用多卡
```

