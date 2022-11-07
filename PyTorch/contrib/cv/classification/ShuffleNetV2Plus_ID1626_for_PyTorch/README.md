#  ShuffleNetV2Plus for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)

# 概述

## 简述
ShuffleNetV2Plus是ShuffleNetV2的增强版本，它在ShuffleNetV2的基础上加入了Hard-Swish、Hard-Sigmoid激活函数以及SE注意力模块，进一步提升了ShuffleNetV2的性能，达到了更高的图像分类准确率。
- 参考实现：

  ```
  url=https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2%2B
  commit_id=d69403d4b5fb3043c7c0da3c2a15df8c5e520d89
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}        # 克隆仓库的代码 
  cd {code_path}         # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，下载开源数据[ImageNet2012](http://www.image-net.org/)训练集和验证集，将数据集上传到服务器任意路径下并解压。

2. 数据预处理。
  按照训练集格式处理验证集，可以使用以下[脚本](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
  以ImageNet2012数据集为例，数据集目录结构参考如下所示。

   ```
   ├── ImageNet2012
         ├──train
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...   
              ├──...                     
         ├──val  
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...       
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...              
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称} 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1p.sh --data_path=/data/xxx/    
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   
     ```

    --data\_path参数填写数据集路径。

    模型训练脚本参数说明如下。  
   ```
   公共参数：
    --data                              //数据集路径
    --workers                           //加载数据进程数      
    --epoch                             //重复训练次数
    --batch-size                        //训练批次大小
    --learning-rate                     //初始学习率
    --momentum                          //动量
    --wd                                //权重衰减
    --addr                              //主机地址
    --port                              //主机端口
    --amp                               //是否使用混合精度
    --loss-scale                        //混合精度lossscale大小
    --eval-freq                         //测试周期
    --multiprocessing-distributed       //是否使用多卡训练
    --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
    --device_num 8                      //启动的卡的数目
   ```
   
   训练完成后，权重文件默认会写入到和test文件同一目录下，并输出模型训练精度和性能信息到网络脚本test下output文件夹内。
   ```
    test/output/devie_id/train_${device_id}.log           # 训练脚本原生日志
    test/output/devie_id/ShuffleNetV2Plus_ID1626_for_PyTorch_bs1024_8p_perf.log  # 8p性能训练结果日志
    test/output/devie_id/ShuffleNetV2Plus_ID1626_for_PyTorch_bs1024_8p_acc.log  # 8p精度训练结果日志
    checkpoint.pth.tar                            # checkpoits
   ```

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | AMP_Type |
| ------- | ----- | ---: | ------ | -------: |
| 1p-NPU1.5 | -     |  2210 | 1      |        - |
| 1p-NPU1.8  | -     |  742 | 1      |       O2 |
| 8p-NPU1.5 | 73.132 | 12861 | 360    |        - |
| 8p-NPU1.8  | 73.398 | 5655 | 360    |       O2 |


# 版本说明

## 变更

2022.07.05：更新torch1.8版本，重新发布。

2020.07.08：首次发布。

## 已知问题

无。
