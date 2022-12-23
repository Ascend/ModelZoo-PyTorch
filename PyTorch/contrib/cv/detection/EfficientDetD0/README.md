# EfficientDet_D0-Pytorch

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

EfficientDet是google在2019年11月发表的一个目标检测算法系列，分别包含了从D0~D7总共八个算法，对于不同的设备限制，能给到SOTA的结果，在广泛的资源约束下始终比现有技术获得更好的效率。特别是在单模型和单尺度的情况下，EfficientDet-D7在COCO测试设备上达到了最先进的52.2AP，具有52M参数和325B FLOPs，相比与之前的算法，参数量缩小了4到9倍，FLOPs缩小了13到42倍。

- 参考实现：

  ```
  url=https://github.com/rwightman/efficientdet-pytorch
  commit_id=c5b694aa34900fdee6653210d856ca8320bf7d4e
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/ModelZoo-PyTorch.git
  code_path=PyTorch/contrib/cv/detection
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
  | 硬件 | [1.0.12](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动 | [21.0.3.1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [5.0.3](https://www.hiascend.com/software/cann/commercial?version=5.0.3) |
  | PyTorch    | [1.5.0](https://gitee.com/ascend/pytorch/tree/v1.5.0/) |

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip install -r requirements.txt
  ```

  timm版本为0.4.12，其中timm源码有修改，需要替换timm中以下文件，修改后的文件可从timm_modify文件夹中复制替换到对应位置：
  - timm
    - ├── models
      - ├── layers
        - ├── conv2d_same.py
        - ├── pool2d_same.py
        - ├── padding.py
        - └── activations_me.py
    - ├── optim
      - ├── optim_factory.py
  - 由于原始timm包存在算子不适配和性能的问题，使用修改后的timm，具体见Code modification。


- Code modification
  - effdet文件中的修改：
    - /anchors.py: 使用npu_multiclass_nms算子，替换原来的nms
    - /data/loader.py: 替换所有.cuda -> .npu
    - /evaluator.py: 修改class CocoEvaluator(Evaluator) ->  def evaluate(self) -> metric dtype = torch.float32
    - /loss.py: 替代one_hot算子为torch.npu_one_hot算子
    - /efficientdet.py： 修改class FpnCombine(nn.Module) -> def forward(self, x: List[torch.Tensor]) -> stack算子

- timm文件中的修改：
  - /models/layers文件夹中针对padv3d算子不适配的情况有三处规避修改,使用自行修改的pad，规避padsame算子：
    - /conv2d_same.py
    - /pool2d_same.py
    - /padding.py
  - /optim/optim_factory.py: 替换optimizers为NpuFusedSGD

## 准备数据集

1. 获取数据集。

   从[link](http://cocodataset.org/)下载coco数据集
    - 2017 Train images(http://images.cocodataset.org/zips/train2017.zip)

    - 2017 Val images(http://images.cocodataset.org/zips/val2017.zip)

    - 2017 Train/Val annotations(http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

    - coco数据集目录结构需满足:

       ```
      coco
         ├── annotations
         ├── train2017
         └── val2017
      ```

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   dos2unix ./test/*.sh
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练
     ```
     bash ./test/train_performance_1p.sh --data_path=/data/xxx/
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_full_8p_0-120.sh --data_path=/data/xxx/

     bash ./test/train_performance_8p.sh --data_path=/data/xxx/
     ```
   - 单机8卡评估

     ```
     bash ./test/train_eval_8p.sh --data_path=/data/xxx/ --pth_path=./model_best.pth.tar
   - finetuning
     ```
     bash ./test/train_finetune_1p.sh --data_path=/data/xxx/ --pth_path=./model_best.pth.tar
     ```
   - 运行在线示例
     ```
     python3.7 demo.py
     ```

   --data_path：数据集路径

   --pth_path：训练过程中生成的权重文件路径。

   模型训练脚本参数说明如下。

   ```bash
    --root               //数据集路径,可自行修改为对应路径的coco数据集
    --resume             //加载模型checkpoint路径，可自行修改为对应路径的模型文件
    --addr               //主机地址
    --model              //使用模型，默认：tf_efficientdet_d0
    --opt                //优化器选择
    --epoch              //重复训练次数
    --batch-size         //训练批次大小
    --lr                 //初始学习率，默认：0.16
    --model-ema          //使用ema
    --sync-bn            //是否使用sync-bn
    --device-list        //多卡训练指定训练用卡 ,8卡：'0,1,2,3,4,5,6,7'
    --lr-noise           //学习率噪声
    --amp                //是否使用混合精度
    --loss-scale         //lossscale大小
    --opt-level          //混合精度类型
   ```

 训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

|   名称    | 性能（fps）| 精度（map）|    BS    |  Epochs  | AMP_Type  |
| :------: | :------:  | :------: | :------: | :------: | :------:  |
| GPU_1P   |    48     | -        | 16       | 1        |     O1    |
| GPU_8P   |   270     | 0.3346   | 16       | 310      |     O1    |
| NPU_1P   |    16     | -        | 16       | 1        |     O1    |
| NPU_8P   |   112     | 0.3289   | 16       | 310      |     O1    |

# 版本说明

## 变更

2020.07.08：首次发布。

## 已知问题
- 代码仓最高精度0.336；NPU8卡最高精度出现在第270epoch，精度0.3289；GPU8卡最高精度出现在第250epoch，精度0.3346。
- 性能优化过程中，发现此模型在NPU上，基本的conv，bn，act算子耗时为GPU的3-5倍，因此优化较难。
- GPU训练时，若采用固定lossscale时会在训练时出现loss为nan的情况，故采用源码提供的动态lossscale，动态调整前期losssacle很小（实测为8），后期lossscale很大（万位级）的情况；NPU训练采用了固定的lossscale。其他参数保持一致。
- NPU_8P训练过程由于设备公用情况的限制，同时与GPU同步，防止lossscale较大出现loss为nan的情况，前130epoch在910B卡上训练，lossscale设置为8，性能较910A卡稍差；后180个epoch在910A卡上训练，lossscale设置为128。









