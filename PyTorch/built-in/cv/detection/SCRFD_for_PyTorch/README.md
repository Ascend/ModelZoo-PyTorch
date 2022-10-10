# SCRFD for PyTorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

SCRFD是通过NAS（神经网络搜索）得到的一个人脸检测模型，通过使用ResNet作为主干网络，融入PAFPN、ATSS Assigner等模块，可以获得更好的精度、更高的性能


- 参考实现：

  ```
  url=https://github.com/deepinsight/insightface.git
  commit_id=babb9a58bbc42ae4b648acdbb803159a35f53db3
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/built-in/cv/detection
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
  | 固件与驱动 | [22.0.RC3.B020](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.1.RC1.B010](https://www.hiascend.com/software/cann/commercial?version=5.1.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)或[1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。
  
- 安装依赖（根据模型需求，按需添加所需依赖）。

  ```
  1. 安装依赖
    pip install -r requirements.txt
    
  2. 安装mmcv（如果环境中有mmcv,请先卸载再执行以下步骤）
    git clone -b v1.4.8 https://github.com/open-mmlab/mmcv.git
    bash tools/build_mmcv.sh

  3. 安装mmdet（如果环境中有mmdet,请先卸载再执行以下步骤）
    pip3.7 install -v -e .
  ```


## 准备数据集

1. 获取数据集。

   用户自行获取原始数据集，可选用的开源数据集包括WIDER_FACE，将数据集上传到服务器任意路径下并解压。

   数据集目录结构参考如下所示。

   ```
   ├── WIDERFace/
         ├──train
              ├──images
                    │──图片1
                    │──图片2
                    │   ...       
              ├──labelv2.txt              
         ├──val  
              ├──images
                    │──图片1
                    │──图片2
                    │   ...       
              ├──gt  
                    │──*.gt    
              ├──labelv2.txt 
   ```

   > **说明：** 
   >该数据集的训练过程脚本只作为一种参考示例。

2. 数据预处理（按需处理所需要的数据集）。

## 获取预训练模型（可选）

请参考原始仓库上的README.md进行预训练模型获取。将获取的bert\_base\_uncased预训练模型放至在源码包根目录下新建的“temp/“目录下。

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
     bash ./test/train_full_1p.sh --data_path=xxx/WIDERFace    
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p.sh --data_path=xxx/WIDERFace
     ```

   --data\_path参数填写数据集路径。

   模型训练脚本参数说明如下。

   ```
   公共参数：
   --config                            //配置文件路径
   --work-dir                          //工作目录
   --no-validate                       //是否禁用训练中的eval流程
   --perf                              //是否进行性能测试      
   --seed                              //随机种子
   --local_rank                        //当前进程的rank号   --world_size                        //全局设备数目
   --master_addr                       //主进程IP地址
   --master_port                       //主进程端口号
   --npuid                             //设备ID偏移
   ```
   
   训练完成后，权重文件保存在工作目录下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME    | Acc@1 |  FPS | Epochs | AMP_Type |
| ------- | ----- | ---: | ------ | -------: |
| 1p-竞品 | -     |  78 | 3      |        - |
| 1p-NPU  | -     |  80 | 3      |       O2 |
| 8p-竞品 | [95.16, 93.87, 83.05] | 278 | 640    |        - |
| 8p-NPU  | [95.02, 93.79, 82.67] | 417 | 640    |       O2 |

备注：一定要有竞品和NPU。

# 版本说明

## 变更

2022.09.20：首次发布。

## 已知问题


目前可能会出现mmpycocotools与pycocotools两个第三方库冲突的问题，如果出现pycocotools相关问题，需要首先将mmpycocotools和pycocotools全部卸载，然后重装mmpycocotools即可解决冲突问题。











