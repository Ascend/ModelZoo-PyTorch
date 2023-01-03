# CSWin-Transformer
- [CSWin-Transformer](#cswin-transformer)
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
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  apt-get install -y numactl
  pip install -r requirements.txt
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
1. 下载一些必要文件。
项目目录下新建dataset文件夹，并在dataset路径下添加文件，文件下载地址：https://github.com/microsoft/CSWin-Transformer/tree/main/dataset

   > **说明：** 
   > 该数据集的训练过程脚本只作为一种参考示例。


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
     bash ./test/train_full_1p.sh --data_path=real_data_path
     ```
    
    - 单机8卡训练
      
      启动单卡训练。
     ```
     bash ./test/train_full_8p.sh --data_path=real_data_path
     ```

   --data_path参数填写数据集路径。


   ```
   公共参数：
    --data_path                         //数据集路径     
    --epochs                            //重复训练次数
    --batch-size                        //训练批次大小
    --lr                                //初始学习率
    --local_rank                        //默认卡号
    --weight-decay                      //权重衰减
    --seed                              //训练的随机数种子 
    --amp                               //是否使用混合精度
   ```

     训练完成后，权重文件默认会写入到output目录下，并输出模型训练精度和性能信息到test下output文件夹内。

# 训练结果展示

**表 2**  训练结果展示表

|  名称  | 精度  | 性能 |  Epochs  | AMP_type  | Torch_version |
| :----: | :---: | :--: | :----: | :---: | :--: |
| GPU-1p |   -   | 230  | 1 |   -   | -  |
| GPU-8p | 82.3  | 1700 | 310 | -  | - |
| NPU-1p |   -   | 276  | 1 |   O2   | 1.5  |
| NPU-1p |   -   | 150  | 1 |   O2   | 1.8  |
| NPU-8p |   82.45  | 2234 | 310 |   O2   | 1.5 |
| NPU-8p |   82.45   | 1210 | 310 |   O2   | 1.8 |



# 版本说明

## 变更

2022.11.23：更新torch1.8版本，重新发布。

2022.01.15：首次发布。

## 已知问题

无。