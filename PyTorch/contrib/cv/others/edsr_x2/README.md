# EDSR_x2 for PyTorch

# 概述

EDSR是一种增强的深度超分辨率网络，通过从传统的ResNet架构中移除不必要的模块，采用因子为0.1的残差尺度来提高模型性能。在每个剩余块中，在最后一个卷积层之后放置常数尺度层。当使用大量的滤波器时，这些模块极大地稳定了训练过程。

- 参考实现：

    ```
    url=https://github.com/thstkdgus35/EDSR-PyTorch 
    commit_id=585ce2c4fb80ae6ab236f79f06911e2f8bef180c
    ```

- 适配昇腾 AI 处理器的实现：

    ```
    url= https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/contrib/cv/others
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

   表 1 版本配套表

    |    配套   |    版本   |
    |----------|---------- |
    | 固件与驱动 |  [5.1.RC2](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial)  |
    |   CANN    |  [5.1.RC2](https://www.hiascend.com/software/cann/commercial?version=5.1.RC2) |
    |  PyTorch  |  [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)   |

- 环境准备指导。

    请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖（根据模型需求，按需添加所需依赖）。

    `pip install -r requirements.txt`

## 准备数据集
1. 获取数据集。

    用户自行获取原始数据集。
    Download the DIV2K dataset from https://cv.snu.ac.kr/research/EDSR/DIV2K.tar (~7.1 GB)。

    ```
   ├── DIV2K
         ├── DIV2K_test_LR_bicubic
              ├──X2     
              ├──X3  
              ├──X4                    
         ├── DIV2K_test_LR_unknown  
              ├──X2      
              ├──X3
              ├──X4
         ├── DIV2K_train_LR_bicubic
              ├──X2      
              ├──X3  
              ├──X4                    
         ├── DIV2K_train_LR_unknown  
              ├──X2    
              ├──X3
              ├──X4 
     ```
2. 数据预处理

    加载数据过程中会自行实现预处理。如有其他需要请自行补充。
# 开始训练
## 训练模型

1. 进入解压后的源码包根目录。

    ```
    cd /${模型文件夹名称}
    ```

2. 运行训练脚本。

    ```bash
    # real_data_path为包含DIV2K数据集文件夹的目录
    
    # 1p training full
    bash test/train_full_1p.sh --data_path=real_data_path
    
    # 1p train perf
    bash test/train_performance_1p.sh --data_path=real_data_path
    
    # 1p testing
    bash test/train_eval_1p.sh.sh --pre_train_model=/path/to/model_best.pt --data_path=real_data_path
    
    # 8p training full
    bash test/train_full_8p.sh --data_path=real_data_path
    
    # 8p train perf
    bash test/train_performance_8p.sh --data_path=real_data_path
    
    # 8p testing
    bash test/train_eval_8p.sh.sh --pre_train_model=/path/to/model_best.pth --data_path=real_data_path
    
    # finetuning
    bash test/train_finetune_1p.sh --data_path=real_data_path --pre_train_path=/path/to/model_best.pt
    
    # demo
    # 先将待测试图片放到 test 文件夹，输出图片会放在在 output_sr 文件夹
    python3 demo.py --cpu --pre_train=/path/to/model_best.pth
    ```
日志路径：

单卡训练

```
test/output/train_${device_id}/train_${device_id}.log  # training detail log
test/output/train_${device_id}/EDSR_x2_bs16_1p_acc  # 1p training performance result log
test/output/train_${device_id}/train_EDSR_x2_bs16_1p_acc_loss   # 1p training accuracy result log
``` 

多卡训练

```
test/output/train_${device_id}/train_${device_id}.log  # training detail log
test/output/train_${device_id}/EDSR_x2_bs16_8p_acc  # 8p training performance result log
test/output/train_${device_id}/train_EDSR_x2_bs16_8p_acc_loss   # 8p training accuracy result log
```


   模型训练脚本参数说明如下，以train_performance_1p.sh为例：

    
    ################基础配置参数，需要模型审视修改##################
    # 必选字段(必须在此处定义的参数): Network batch_size RANK_SIZE
    # 网络名称，同目录名称
    Network="EDSR_x2"
    # 训练batch_size
    batch_size=16
    # 训练使用的npu卡数
    export RANK_SIZE=1
    # 数据集路径,保持为空,不需要修改 train_path=./data/TrainDataset
    data_path=""

    # 训练epoch
    train_epochs=2
    # 指定训练所使用的npu device卡id=0 多卡id=1(用于输出日志)
    device_id=0
    # 加载数据进程数
    workers=128

    公共参数：
    --data_path=./DIV2K                     //训练数据集路径 
    --seed=49                               //设置随机种子
    --workers=${workers}                    //加载数据进程数
    --lr=1e-4                               //初始学习率
    --world-size=1                          //服务器台数
    --device='npu'                          //计算芯片类型
    --gpu=${ASCEND_DEVICE_ID}               //计算芯片序号
    --dist-backend='hccl'                   //通信后端
    --epoch=${train_epochs}                 //训练epoch数
    --loss-scale=128                        //loss-scale大小
    --amp                                   //是否开启混合精度
    --batchsize=${batch_size} > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &                        //日志路径

# 训练结果展示
表 2 训练结果展示表

| NAME    | PT版本|PSNR(dB) |  FPS | Epochs | AMP_Type |
| ------- |---- |----- | ---: | ------ | -------: |
| 1p-竞品V | 1.5|-     |- | -      |        - |
| 1p-NPU  | 1.5|35.001     |- | 281      |       O2 |
| 1p-NPU  | 1.8|35.866     |32.305| 86      |       O2 |
| 8p-竞品V | 1.5|- |- | -    |        - |
| 8p-NPU  | 1.5|34.943 |- | 300    |       O2 |
| 8p-NPU  | 1.8|34.802 |287.435 | 86    |       O2 |

# 版本说明
## 变更

2022.08.15：更新pytorch1.8版本，重新发布。

2020.07.08：首次发布。

## 已知问题
无
