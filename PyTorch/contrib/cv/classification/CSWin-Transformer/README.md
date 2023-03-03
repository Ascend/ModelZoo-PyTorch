# CSWin-Transformer for PyTorch

- [概述](#概述)
- [准备训练环境](#准备训练环境)
- [开始训练](#开始训练)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)


# 概述
## 简述

CSWin-Transformer提出的 CSWin 是使用了条状区域来计算 attention ，在网络的不同阶段使用不同宽度的条状区域，在节约计算资源的同时实现了强大的特征建模能力。

- 参考实现：
  ```
  url=https://github.com/microsoft/CSWin-Transformer
  commit_id=f111ae2f771df32006e7afd7916835dd67d4cb9d
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

 - 安装 `numactl` 。
   ```
   apt-get install -y numactl
   ```

## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括ImageNet，将数据集上传到服务器任意路径下并解压。
   
   以ImageNet数据集为例，数据集目录结构参考如下所示。

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

2. 数据预处理。

   在源码包根目录下新建 `dataset` 文件夹，并在 `dataset` 路径下添加 `ILSVRC2012_name_train.txt` 、`ILSVRC2012_name_val.txt` 和 `imagenet_class_index.json` 文件，文件下载地址：https://github.com/microsoft/CSWin-Transformer/tree/main/dataset 。


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
     bash ./test/train_full_8p.sh --data_path=/data/xxx/  # 8卡精度
     
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/  # 8卡性能
     ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   模型训练脚本参数说明如下。

   ```
   公共参数：
    --data                         //数据集路径     
    --model                        //模型名称
    -b                             //训练批次大小
    --lr                           //初始学习率
    --local_rank                   //训练设备卡号
    --weight-decay                 //权重衰减
    --img-size                     //训练图片大小 
    --amp                          //是否使用混合精度
    --warmup-epochs                //预热重复训练次数
    --model-ema-decay              //ema衰减
    --drop-path                    //随机路径删除参数
   ```

   训练完成后，权重文件默认会写入到output目录下，并输出模型训练精度和性能信息到test下output文件夹内。

# 训练结果展示

**表 2**  训练结果展示表

|   NAME   | Acc@1 | FPS  | Epochs | AMP_Type | Torch_Version |
| :------: | :---: | :--: | :----: | :------: | :-----------: |
| 1p-竞品V |   -   | 230  | 1 |   -   | 1.5  |
| 8p-竞品V | 82.3  | 1700 | 310 | -  | 1.5 |
| 1p-NPU |   -   | 150  | 1 |   O2   | 1.8  |
| 8p-NPU |   82.45   | 1210 | 310 |   O2   | 1.8 |


# 版本说明

## 变更

2022.11.23：更新torch1.8版本，重新发布。

2022.01.15：首次发布。

## FAQ

无。