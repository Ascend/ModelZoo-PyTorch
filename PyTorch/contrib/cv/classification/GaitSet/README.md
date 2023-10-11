# GaitSet for PyTorch

- [概述](概述.md)
- [准备训练环境](准备训练环境.md)
- [开始训练](开始训练.md)
- [训练结果展示](训练结果展示.md)
- [版本说明](版本说明.md)

# 概述

## 简述

GaitSet是一个灵活、有效和快速的跨视角步态识别网络。灵活性：其输入可以是轮廓组成的集合，并没有其他约束。有效性：它在CASIA-B数据集上达到95.0%的准确率。快速性：使用8个NVIDIA 1080TI GPU，它只需7分钟即可在OU-MVLP数据集上完成评估，这个数据集包含13w序列，平均每条序列有70个图像。

- 参考实现：

  ```
  url=https://github.com/AbnerHqC/GaitSet
  commit_id=5535943428b66415530d8379b648b8f74a294219
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
  | PyTorch 1.11   | numpy==1.22.1 |
  | PyTorch 2.1   | - |
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  在模型源码包根目录下执行命令，安装模型需要的依赖。
  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集
   
    请用户自行下载`CASIA-B`数据集，将数据集上传到服务器任意路径下并解压，解压后数据集内部的目录应为（`CASIA-B`数据集）：数据集路径/对象序号/行走状态/角度，例如`CASIA-B/001/nm-01/000/ `。

   `CASIA-B`数据集目录结构参考如下所示。

   ```
   ├── CASIA-B
         ├──001
              ├──bg-01
                    │──000
                        │──图片1
                        │──图片2
                        │   ...  
                    │──018
                        │──图片1
                        │──图片2
                        │   ... 
                    │   ...       
              ├──cl-01
                     │──000
                        │──图片1
                        │──图片2
                        │   ...  
                     │──018
                        │──图片1
                        │──图片2
                        │   ... 
                     │   ...    
              ├──nm-01
                     │──000
                        │──图片1
                        │──图片2
                        │   ...  
                     │──018
                        │──图片1
                        │──图片2
                        │   ... 
                     │   ...    
              ├──...                     
         ├──002  
              ├──bg-01
                    │──000
                        │──图片1
                        │──图片2
                        │   ...  
                    │──018
                        │──图片1
                        │──图片2
                        │   ... 
                    │   ...       
              ├──cl-01
                     │──000
                        │──图片1
                        │──图片2
                        │   ...  
                     │──018
                        │──图片1
                        │──图片2
                        │   ... 
                     │   ...    
              ├──nm-01
                     │──000
                        │──图片1
                        │──图片2
                        │   ...  
                     │──018
                        │──图片1
                        │──图片2
                        │   ... 
                     │   ...    
              ├──... 
         ├──...     
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

3. 数据预处理。

    在源码包根目录下执行一下命令，使用`pretreatment.py`进行数据处理：其中括号"{}"需要替换为数据集的路径。
    
    ```bash
    # --input_path为原数据集‘CASIA-B’的路径； --output_path为预处理后的数据集路径。
    python3 pretreatment.py --input_path {downloaded_path} \
                            --output_path {output_path}
    ```

    >  预处理过程中提示`--WARNING--`属于预期现象，请等待处理完成

# 开始训练

## 训练模型


1. 进入解压后的源码包根目录。

    ```bash
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

   - 单机8卡评测

     启动8卡评测。

     ```
     bash ./test/train_eval_8p.sh --data_path=/data/xxx/
     ```
     > **注意：** 需要手动把`train_eval_8p.sh`中`train_iters`参数改为训练保存的模型想要加载的代数。

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   --iters参数填写模型训练的迭代次数。

   --hf32开启HF32模式，不与FP32模式同时开启

   --fp32开启FP32模式，不与HF32模式同时开启

   模型训练脚本参数说明如下。
   
   ```
   公共参数：
   --data_path                         //数据集路径
   --addr                              //主机地址
   --iters                             //迭代次数
   --port                              //主机端口
   --world_size                        //分布式训练节点数
   --dist_backend                      //通信后端
   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示
**表 3** 训练结果展示表

| NAME | 精度(RANK-1, %) | 性能(FPS) | 训练代数(Iters)  | AMP_Type | Torch_Version |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1P_NPU-ARM | - | 255.138 |  6w | O2 | 1.8       |
| 8P_NPU-ARM | 93.744 | 826.016 |  4w | O2 | 1.8       |
| 1P_NPU-非ARM | - | 278.897 |  6w | O2 | 1.8       |
| 8P_NPU-非ARM | 93.744 | 1428.763 |  4w | O2 | 1.8       |

# 版本说明

## 变更

2023.03.08：更新readme，重新发布。

## FAQ

无。
