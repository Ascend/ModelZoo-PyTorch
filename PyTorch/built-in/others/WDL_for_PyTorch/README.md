# Wide&Deep for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

Wide&Deep是一个同时具有Memorization和Generalization功能的CTR预估模型，该模型主要由广义线性模型（Wide网络）和深度神经网络（Deep网络）组成，对于推荐系统来说，Wide线性模型可以通过交叉特征转换来记忆稀疏特征之间的交互，Deep神经网络可以通过低维嵌入来泛化未出现的特征交互。与单一的线性模型（Wide-only）和深度模型（Deep-only）相比，Wide&Deep可以显著提高CTR预估的效果，从而提高APP的下载量。

- 参考实现：

  ```
  url=https://github.com/shenweichen/DeepCTR-Torch.git
  commit_id=8265c75237e473c7f238fd6ba44cb09f55d1d9a9
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/others
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
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/) |

- 所需依赖。

  **表 2**  依赖列表
  
  | 依赖名称    | 版本             |
  | ----------- | ---------------- |
  | torch       | 1.8.1+ascend.rc2 |
  | torch-npu   | 1.8.1rc2         |
  | Pillow      | 9.2.0            |
  | apex        | -                |
  | torchvision | 0.2.2.post2           |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖。

  ```
  pip install -r requirements.txt
  ```

## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，下载[criteo数据集](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)，将数据集上传到服务器任意路径下并解压。

   解压后的目录结构如下：

   ```
   ├── criteo
         ├──criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz # 原始数据集文件                 
         ├──train.txt                                                           # 解压后生成
         ├──test.txt                                                            # 解压后生成
         ├──readme.txt                                                          # 解压后生成       
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理。
- 将源码包根目录下的criteo_preprocess.py拷贝到数据集路径
- 进入数据集目录并执行 python3 criteo_preprocess.py train.txt
- 执行上述脚本后，在当前路径下生成 wdl_trainval.pkl, wdl_test.pkl, wdl_infer.txt

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

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --seed                              //初始化训练的种子，默认：1234
   --device_id                         //设备id，默认：0  
   --rank                              //分布式训练的节点等级，默认：0
   --dist                              //8卡分布式训练，默认：False
   --device_num                        //用于训练的npu设备数量，默认：1
   --amp                               //是否使用混合精度，默认：False
   --loss_scale                        //混合精度lossscale大小
   --opt_level                         //apex opt level，默认：01
   --data_path                         //数据集路径
   --resume                            //最新checkpoint的路径，默认值：无
   --checkpoint_save_path              //保存最新checkpoint的路径，默认值：./(当前路径)
   --lr                                //初始学习率，默认：0.0001
   --batch_size                        //训练批次大小，默认：1024
   --eval_batch_size                   //测试批次大小，默认：16000
   --epochs                            //重复训练次数，默认：3
   --start_epoch                       //开始训练记录，默认：0
   --sparse_embed_dim                  //The embedding dims for sparse features，默认：4
   --steps                             //训练步数，默认：0

   ```
   
   训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息（保存在output目录下）。

# 训练结果展示

**表 3**  训练结果展示表


| NAME    | Acc@1  | FPS     | Epochs | AMP_Type | Torch_version |
| ------- | ------ | :------ | ------ | :------- | :------------ |
| 1p-NPU  | -      | 51154.6454 | 3     | O1       | 1.5           |
| 1p-NPU  | -      | 51883.6480 | 3     | O1       | 1.8           |
| 8p-NPU  | 0.7978 | 171545.5429 | 3     | O1       | 1.5           |
| 8p-NPU  | 0.7961 | 214769.5062 | 3     | O1       | 1.8           |


# 版本说明

## 变更

2022.08.17：更新torch1.8版本，重新发布。

2021.03.18：首次发布。

## 已知问题

无。
