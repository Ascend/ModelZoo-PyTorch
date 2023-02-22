# BEiT for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

BEiT是一种自监督视觉表示模型，提出了一种用于预训练视觉Transformer的masked image modeling任务，主要目标是基于损坏的图像patch块恢复原始视觉token。

- 参考实现：

  ```
  url=https://github.com/microsoft/unilm.git
  commit_id=006195f51b10ac44773cb62bad854fdfebb3c6c8
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

   用户自行下载 `ImageNet2012` 开源数据集，将数据集上传到服务器任意路径下并解压。
    
   数据集目录结构参考如下所示。

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

## 获取预训练模型
    
用户自行获取预训练模型，将获取的 `beit_base_patch16_224_pt22k_ft22k.pth` 权重文件放至在源码包根目录下新建的 `checkpoints` 目录下。

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
     bash ./test/train_full_1p.sh --data_path=/data/xxx/  # 单卡精度  
     
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  # 单卡性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   # 8卡精度

     bash ./test/train_performance_8p.sh --data_path=/data/xxx/    # 8卡性能
     ```
   - 单机单卡评测

     启动单卡评测。

     ```
     bash ./test/train_eval_1p.sh --data_path=/data/xxx/ --resume=XXX # 单卡评测
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。
   
   --resume参数填写训练权重生成路径，需写到权重文件的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --model                             //模型名称
   --data_path                         //数据集路径
   --finetune                          //是否微调
   --output_dir                        //输出目录
   --batch_size                        //训练批次大小
   --lr                                //初始学习率
   --update_freq                       //更新频率
   --warmup_epochs                     //热身训练轮次
   --epochs                            //重复训练次数
   --layer_decay                       //层衰减
   --drop_decay                        //丢失衰减    
   --weight_decay                      //权重衰减
   --nb_classes                        //类别数量
   --amp                               //是否使用混合精度
   --device                            //设备
   --opt-level                         //混合精度类型
   多卡训练参数：
   ----nproc_per_node                  //训练使用卡的数量

   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V | - | - | 1  | - | 1.5 |
| 8p-竞品V | - | - | 90 | - | 1.5 |
| 1p-NPU  | -     |  149 | 1      |       O2 |1.8    |
| 8p-NPU  | 85.238 | 1157 | 30    |       O2 |1.8    |


# 版本说明

## 变更

2022.10.24：首次发布。

## FAQ

无。

