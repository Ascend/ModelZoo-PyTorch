# Evo-Levit for PyTorch
-   [概述](#概述)
-   [准备训练环境](#准备训练环境)
-   [开始训练](#开始训练)
-   [训练结果展示](#训练结果展示)
-   [版本说明](#版本说明)

# 概述

## 简述

Evo-ViT的具体框架设计，包括基于全局class attention的token选择以及慢速、快速双流token更新两个模块。其根据全局class attention的排序判断高信息token和低信息token，将低信息token整合为一个归纳token，和高信息token一起输入到原始多头注意力（Multi-head Self-Attention, MSA）模块以及前向传播（Fast Fed-forward Network, FFN）模块中进行精细更新。更新后的归纳token用来快速更新低信息token。全局class attention也在精细更新过程中进行同步更新变化。

- 参考实现：

  ```
  url=https://github.com/YifanXu74/Evo-ViT
  commit_id=4c5d9b30b0a3c9b1e7b8687a9490555bd9d714ca
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
  | 固件与驱动 | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial ) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1 ) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install timm==0.4.12
  pip install torchvision==0.9.1
  pip install torch_npu-1.8.1rc2.20220607-cp37-cp37m-linux_aarch64.whl
  pip install torch-1.8.1+ascend.rc2.20220607-cp37-cp37m-linux_aarch64.whl
  pip install apex-0.1+ascend.20220607-cp37-cp37m-linux_aarch64.whl
  ```
  
- 关于timm包的NPU优化补丁。

  ```
  # 需要先cd到当前文件目录，一般timm包的安装位置在/usr/local/lib/python3.7/dist-packages/timm/
  #先后生成补丁并升级包
  diff -uN {timm_path}/data/mixup.py {code_path}/fix_timm/mixup.py >mixup.patch
  diff -uN {timm_path}/optim/optim_factory.py {code_path}/fix_timm/optim_factory.py >optim.patch
  patch -p0 {timm_path}/data/mixup.py mixup.patch
  patch -p0 {timm_path}/optim/optim_factory.py optim.patch
  ```

  


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集ImageNet2012，将数据集上传到服务器任意路径下并解压。

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
   > 数据集路径以用户自行定义的路径为准

## 获取预训练模型

Evo-Vit模型训练需要配置teacher—model，获取方式为在GitHub的[Evo-Vit]([GitHub - YifanXu74/Evo-ViT: Official implement of Evo-ViT: Slow-Fast Token Evolution for Dynamic Vision Transformer](https://github.com/YifanXu74/Evo-ViT)),checkpoint文件可以在该仓库自行下载，也可以直接使用网址进行下载，网址如下
https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth

预训练模型需要放置在模型文件夹下，与main_levit.py或者README处于同级目录下。与源码中的配置参数的默认值 ”./regnety_160-a5fe301d.pth“保持一致。

# 开始训练

## 训练模型
1. 进入解压后的源码包根目录。

    ```
    cd /${模型文件夹名称} 
    ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练，开始训练前，请用户根据实际路径配置data_path参数。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_full_1P.sh --data_path=/data/xxx/    
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8P.sh --data_path=/data/xxx/   
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --data_path                         //数据集路径     
   --epochs                            //重复训练次数
   --batch-size                        //训练批次大小
   --nproc_per_node                    //数字表示启用单卡还是多卡
   ```
   
   训练完成后，权重文件保存在当前路径的save中，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME   | PT版本 |  精度 | FPS    | Epochs | AMP_Type |
| ------ | ------ | ----: | ------ | -----: | -------- |
| 1P-GPU | 1.8.1  |     - | 51     |      1 | O1       |
| 1P-NPU | 1.8.1  |     - | 66.93  |      1 | O1       |
| 8P-GPU | 1.8.1  | 73.54 | 487    |    100 | O1       |
| 8P-NPU | 1.8.1  | 74.32 | 510.72 |    100 | O1       |


# 版本说明

## 变更

2022.11.09：首次发布。

## 已知问题

无。