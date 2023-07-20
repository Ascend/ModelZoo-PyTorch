# SENet154

This implements training of SENet154 on the ImageNet dataset. 

Code of SENet is mainly migrated and adjusted from [GitHub](https://github.com/Cadene/pretrained-models.pytorch#senet).

## SENet154 Detail 

SENet involves group convolution, which may cause error on NPU platforms where group convolution is not well-supported.

Label smoothing is required for qualified model accuracy.

## Requirements
- pytorch_ascend, apex_ascend
- munch package, which can be installed via `pip install munch`
- Download the ImageNet dataset 
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

### 单卡训练流程
1. 安装环境
2. 修改参数
    1. `--data DIR`：ImageNet数据集的存储目录，训练集与验证集分别位于DIR/train和DIR/val
    1. `--log-file FILENAME`：自定义日志文件名
    2. `--device DEVICE`：所使用的单卡训练设备，如cuda:0或npu:0
    3. `--opt-level L`：apex混合精度优化等级，支持O2（默认）或O1
    4. `--loss-scale S`：apex混合精度使用的loss scale，默认为128
    5. `--scheduler`：训练使用的学习率调整器，支持`step`（对应StepLR）和`cosine`（对应CosineAnnealingLR）
3. 开始训练
    ```
   bash ./test/train_full_1p.sh  --data_path=数据集路径    # 精度训练
   bash ./test/train_performance_1p.sh  --data_path=数据集路径  # 性能训练
   ```

### 多卡训练流程
1. 安装环境
2. 修改参数
    1. `--device DEVICE`：所使用的多卡训练设备类别，支持cuda和npu
    2. `--distributed`：开启分布式训练模式
    3. `--num-devices N`：参与训练的设备个数，设备ID依次为DEVICE:0 ... DEVICE:(N-1)
    4. `--batch-size N`：分配个每个设备的batch大小
3. 开始训练
   ```
   bash ./test/train_full_8p.sh  --data_path=数据集路径    # 精度训练
   bash ./test/train_performance_8p.sh  --data_path=数据集路径  # 性能训练
   ```

### 训练结果
日志保存在 ./test/output/device-id 路径下

最终训练模型输出至./model.pth，训练过程中生成的存档点位于./models文件夹下

Profile结果输出至./output.prof

## SENet154 Training Result
$E$为当前一轮的Epoch序号，从0开始

### GPU 8p
|Epochs|Learning rate                         |Optimization type|FPS    |Acc@1 |Acc@5 |
|:----:|:------------------------------------:|:---------------:|:-----:|:----:|:----:|
|120   |$0.6\times 0.1^{\lfloor E/30 \rfloor}$|O2               |955.433|79.130|94.058|
|120   |$1\times 0.45^{\lfloor E/10 \rfloor}$ |O2               |954.725|78.341|93.945|
|120   |$0.6\times 0.93^{E}$                  |O2               |949.309|78.100|94.010|
|120   |$0.3\times (1+\cos{\frac{E\pi}{120}})$|O2               |951.374|80.161|94.879|

### NPU 8p
|Epochs|Learning rate                         |Optimization type|FPS     |Acc@1 |Acc@5 |
|:----:|:------------------------------------:|:---------------:|:------:|:----:|:----:|
|120   |$0.3\times (1+\cos{\frac{E\pi}{120}})$|O2               |1022.920|80.564|94.428|

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md