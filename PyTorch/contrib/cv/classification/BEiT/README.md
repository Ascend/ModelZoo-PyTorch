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





## 获取预训练模型
    
用户自行获取[预训练模型](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22k.pth)。将获取好的“beit_base_patch16_224_pt22k_ft22k.pth”权重文件上传到在源码包根目录下新建的“./checkpoints”目录下。

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
     
     #test/train_performance_1p，单p上训练60个step，运行时间为34秒,输出性能日志./output/0/train_0_perf.log
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/  
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     #test/train_full_8p，8p上训练30个epoch，运行时间大约为11小时,输出精度日志./output/0/train_0.log
     bash ./test/train_full_8p.sh --data_path=/data/xxx/   

     #test/train_performance_8p，8p上训练60个step，运行时间为37秒,输出性能日志./output/0/train_0_perf.log
     bash ./test/train_performance_8p.sh --data_path=/data/xxx/    # training performance
     ```
   - 单机单卡验证

     启动单卡验证。

     ```
     # 参考示例：bash ./test/train_eval_1p.sh --data_path=./data/imagenet --resume=./checkpoints/beit_base_patch16_224_pt22k_ft22kto1k.pth
     bash ./test/train_eval_1p.sh --data_path=/data/xxx/ --resume=XXX
     ```

   --data_path参数填写数据集路径；--resume为生成的权重文件路径，请用户根据实际修改。

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
   --epoch                             //重复训练次数
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

| NAME    | Acc@1 |  FPS | Epochs | AMP_Type | s/per step | PyTorch版本 |
| ------- | ----- | ---: | ------ | -------: |-------:| ------- |
| 1p-NPU | -     |  162 | 1      |        O2 |   0.414 |1.5    |
| 1p-NPU  | -     |  149 | 1      |       O2 |    0.425    |1.8    |
| 8p-NPU | 85.279 | 1210 | 30    |        O2 |  0.422 |1.5    |
| 8p-NPU  | 85.238 | 1157 | 30    |       O2 |     0.450   |1.8    |


# 版本说明

## 变更

2022.10.24：首次发布。

## 已知问题

无。

