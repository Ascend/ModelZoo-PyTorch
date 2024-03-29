# Vit_base_patch32_224

-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述
该模型将自然语言处理中的注意力模型移植到图像识别中，切割输入图片并加入位置嵌入，从而得到多个向量输入，然后将多个注意力模块和感知层结合，最后利用输出的class token得到特征向量，并使用感知层进行分类。

- 参考实现：

  ```
  url=https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
  commit_id=20b2d4b69dae2ec185a77a50cf1d38d55d94b657
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
  | PyTorch 1.5 | - |
  | PyTorch 1.8 | - |
  | PyTorch 1.11   | - |  
  | PyTorch 2.1   | - |  

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
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
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   --data\_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --amp                               //是否使用混合精度
   --addr                              //主机地址
   --seed                              //训练的随机数种子   
   --workers                           //加载数据进程数
   --learning-rate                     //初始学习率
   --momentum                          //动量
   --weight-decay                      //权重衰减
   --print-freq                        //打印周期
   --device                            //使用npu还是gpu
   --dist-backend                      //通信后端
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --gpu                               //使用单卡id
   --rank                              //分布式训练节点编号
   --dist-url                          //启用分布式训练网址
   --multiprocessing-distributed       //是否使用多卡训练
   --world-size                        //分布式训练节点数量
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1  |   FPS    | Npu_nums | Epochs | AMP_Type | Torch_Version |
| :------: | :----: | :------: | :------: | :----: | :------: | :-----------: |
| 1p-竞品V |   -    |   122    |    1     |   1    |    O1    |      1.5      |
| 8p-竞品V | 80.772 |   5207   |    8     |   8    |    O1    |      1.5      |
|  1p-NPU  |   -    | 614.548  |    1     |   1    |    O1    |      1.8      |
|  8p-NPU  | 80.55  | 4497.745 |    8     |   8    |    O1    |      1.8      |

# 版本说明

## 变更

2022.02.13：更新readme，重新发布。

2021.09.08：首次发布。

## FAQ

无。
