# ResNet34_ID1594_for_PyTorch


-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)


# 概述

## 简述

ResNet34网络是由来自Microsoft Research的4位学者提出的卷积神经网络，在2015年的ImageNet大规模视觉识别竞赛中获得了图像分类和物体识别的优胜。 残差网络的特点是容易优化，并且能够通过增加相当的深度来提高准确率。其内部的残差块使用了跳跃连接（shortcut），缓解了在深度神经网络中增加深度带来的梯度消失问题。本文使用PyTorch实现了使用ResNet34训练imagenet数据集的具体实例。

- 参考实现：

    ```
    url=https://github.com/pytorch/examples.git
    commit_id=49e1a8847c8c4d8d3c576479cb2fe2fd2ac583de
    ```

- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/contrib/cv/others
    ```

- 通过Git获取代码方法如下：

    ```
    git clone {url}       # 克隆仓库的代码
    cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

   表 1 版本配套表

    |    配套   |    版本   |
    |----------|---------- |
    | 固件与驱动 |  [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
    |   CANN    |  [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
    |  PyTorch  |  [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)   |

- 环境准备指导。

    请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，将数据集上传到服务器任意路径下并解压。

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



   模型训练脚本参数说明如下：

    ```
    公共参数：
    --a                                 //使用模型
    --addr                              //主机地址
    --seed                              //训练的随机数种子   
    --workers                           //加载数据进程数
    --learning-rate                     //初始学习率
    --mom                               //动量
    --weight-decay                      //权重衰减
    --print-freq                        //打印周期
    --device                            //使用npu还是gpu
    --dist-backend                      //通信后端
    --epochs                            //重复训练次数
    --batch-size                        //训练批次大小
    --label-smoothing                   //标签平滑
    --amp                               //是否使用混合精度
    ```
    
    训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

    

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | AMP_Type | Torch_version |
| ------- |----- | ---: | ------ | ------- | -------: |
| 1p-竞品 | -   |972   | 1      |        - | - |
| 1p-NPU  | -   |2160.449 | 1      |       O2 | 1.5 |
| 1p-NPU  | -   |2386| 1      |       O2 | 1.8 |
| 8p-竞品 | 73.4  |6964 | 130   |        - | - |
| 8p-NPU  | 73.347 |11669 | 130    |       O2 | 1.5 |
| 8p-NPU  | 73.25 |13863.422 | 130    |       O2 | 1.8 |

# 版本说明

## 变更

2022.07.14：更新pytorch1.8版本，并发布。

2021.09.10：首次发布。

## 已知问题
无。
