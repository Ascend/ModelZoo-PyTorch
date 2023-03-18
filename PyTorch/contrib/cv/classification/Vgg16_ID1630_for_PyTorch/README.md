# VGG16 for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

VGG由5层卷积层、3层全连接层、softmax输出层构成，层与层之间使用max-pooling分开，所有隐层的激活单元都采用ReLU函数。VGG模型是2014年ILSVRC竞赛的第二名，第一名是GoogLeNet。但是VGG模型在多个迁移学习任务中的表现要优于GoogLeNet。而且，从图像中提取CNN特征，VGG模型是首选算法。它的缺点在于，参数量有140M之多，需要更大的存储空间。

- 参考实现：

  ```
  url=https://github.com/pytorch/examples.git
  commit_id=507493d7b5fab51d55af88c5df9eadceb144fb67
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
  ```


# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1**  版本支持表

  | Torch_Version      | 三方库依赖版本                                 |
  | :--------: | :----------------------------------------------------------: |
  | PyTorch 1.5 | torchvision==0.2.2.post3 |
  | PyTorch 1.8 | torchvision==0.9.1 |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应PyTorch版本需要的依赖。
  ```
  pip install -r 1.5_requirements.txt  # PyTorch1.5版本
  
  pip install -r 1.8_requirements.txt  # PyTorch1.8版本
  ```
  > **说明：** 
  >只需执行一条对应的PyTorch版本依赖安装命令。


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet2012，CIFAR-10等，将数据集上传到服务器任意路径下并解压。

   VGG16训练使用到的ImageNet2012数据集目录结构参考如下所示。

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
     bash ./test/train_full_1p.sh --data_path=xxx # 单卡精度
     bash ./test/train_performance_1p.sh --data_path=xxx  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=xxx # 8卡精度
     bash ./test/train_performance_8p.sh --data_path=xxx  # 8卡性能
     ```

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path=xxx # 8卡评测
     ```

   - 多机多卡性能数据获取流程
     ```
     1. 安装环境
     2. 开始训练，每个机器所请按下面提示进行配置
            bash ./test/train_performance_multinodes.sh --data_path=数据集路径 --batch_size=单卡batch_size*单机卡数 --nnodes=机器总数量 --node_rank=当前机器rank(0,1,2..) --local_addr=当前机器IP(需要和master_addr处于同一网段) --master_addr=主节点IP
     ```
   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data                              //数据集路径
   --j                                 //加载数据进程数
   --epochs                            //重复训练次数
   --start-epoch                       //开始的轮数
   --batch-size                        //批大小
   --lr                                //学习率
   --momentum                          //动量，默认：0.9
   --wd                                //权重衰减，默认：0.0001
   --p                                 //打印频率
   --resume                            //checkpoint的路径
   --e                                 //是否在验证集上的评估模型
   --seed                              //使用随机数种子初始化，默认：49
   --gup                               //使用GPU的ID
   --world-size                        //分布式训练节点数
   --rank                              //进程编号，默认：-1
   --dist-url                          //用于设置分布式训练的url
   --dist-backend                      //分布式后端
   --addr                              //主机地址
   --amp                               //是否使用混合精度
   --opt-level                         //混合精度类型
   --loss-scale-value                  //混合精度lossscale大小
   --device                            //使用设备为GPU或者是NPU
   --stop-step-num                     //在指定stop-step数后终止训练任务
   --eval-freq                         //测试打印间隔        
   多卡训练参数：
   --multiprocessing-distributed       //是否使用多卡训练
   --device-list '0,1,2,3,4,5,6,7'     //多卡训练指定训练用卡
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表
| NAME  | Acc@1  | FPS  | Epochs | AMP_Type | Torch_Version |
|:-----:|:------:|:----:|:------:|:--------:| :-----------: |
| 1P-竞品V| - | - | 90 | - | 1.5 |
| 8P-竞品V| - | - | 90 | - | 1.5 |
| 1p-NPU  | 73.385  | 1024.02  | 90  | O1  | 1.8 |
| 8p-NPU  | 72.949  | 6540.157 | 90  | O1  | 1.8 |


# 版本说明

## 变更

2022.08.22：更新内容，重新发布。

2022.03.18：首次发布。

## FAQ

无。
