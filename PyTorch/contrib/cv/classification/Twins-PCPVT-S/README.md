# Twins-PCPVT-S for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

Twins-PCPVT使用了CPVT中的 conditional position encoding（条件位置编码CPE）来替代PVT中的绝对位置编码，可以在分类和下游任务上可以直接获得大幅的性能提升，尤其是在稠密任务上，由于条件位置编码 CPE 支持输入可变长度，使得视觉 Transformer 能够灵活处理来自不同空间尺度的特征。

- 参考实现：

  ```
  url=https://github.com/Meituan-AutoML/Twins.git
  commit_id=4700293a2d0a91826ab357fc5b9bc1468ae0e987
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/classification
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

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动 | [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```


## 准备数据集

1. 获取数据集。

   下载开源数据集包括ImageNet2012，将数据集上传到服务器任意路径下并解压。
    
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



# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}/test 
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     # training 1p performance
     nohup bash train_performance_1p.sh --data_path=/data/xxx/ 
     
     # finetuning 1p
     nohup bash train_finetune_1p.sh --data_path=/data/xxx/ --finetune_pth=real_checkpoint_path

     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     # training 8p performance
     nohup bash train_performance_8p.sh --data_path=/data/xxx/ 
    
     # training 8p accuracy
     nohup bash train_full_8p.sh --data_path=/data/xxx/ 

     ```

   --data\_path参数填写数据集路径; --finetune\_pth参数填写实际生成的权重文件路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --model                             //模型名称
   --device                            //设备
   --batch_size                        //训练批次大小
   --data_path                         //数据集路径
   --dist-eval                         //多卡验证
   --drop-path                         //失活率
   --epoch                             //重复训练次数
   --finetune                          //是否微调
   --output_dir                        //输出目录
   --lr                                //初始学习率
   --warmup_epochs                     //热身训练轮次
   --weight_decay                      //权重衰减
   多卡训练参数：
   ----nproc_per_node                  //训练使用卡的数量

   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | PyTorch版本 |
| ------- | ----- | ---: | ------ | ------ | 
| 1p-NPU | -     |  292.89 | 1      | 1.5     | 
| 1p-NPU  | -     |  252.66 | 1      | 1.8     | 
| 8p-NPU | 72.79 | 1940.73| 100    | 1.5     | 
| 8p-NPU  | 77.51 | 1934.00| 100    | 1.8     | 


# 版本说明

## 变更

2022.10.24：首次发布。

## 已知问题

无。



