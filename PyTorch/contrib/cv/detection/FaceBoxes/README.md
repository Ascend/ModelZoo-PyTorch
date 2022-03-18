# FaceBoxes

本项目实现了FaceBoxes从GPU到NPU上训练的迁移，源开源代码仓[FaceBoxes.Pytorch](https://github.com/zisianw/FaceBoxes.PyTorch)

## FaceBoxes Detail

本项目对于FaceBoxes.Pytorch做出了如下更改：

1. 将设备从Nvidia GPU迁移到Huawei NPU上。
2. 在源代码的基础上添加了Apex混合精度进行优化。
3. 在模型迁移到NPU上后一些不支持或性能较低的算子放到CPU上进行规避。
4. 针对测试结果添加了新的评估脚本。

## Requirements

```bash
pip install -r requirements.txt
```

- NPU 配套的run包安装
- Python3.7.5
- PyTorch（NPU版本）
- Apex（NPU版本）

### 导入环境变量

```bash
source scripts/npu_set_env.sh
```

### 编译

```bash
git clone https://github.com/Levi0223/FDDB_Evaluation.git
cd FDDB_Evaluation
python3 setup.py build_ext --inplace
mv ../convert.py ../split.py ./
```

### 准备数据集
数据集下载参考源代码仓

1. 下载[WIDER_FACE](https://github.com/zisianw/FaceBoxes.PyTorch)数据集，将图片放在这个目录下（数据集包含32203张图片）：

   ```bash
   $FaceBoxes_ROOT/data/WIDER_FACE/images/
   ```

   下载转换后的[标注文件](https://github.com/zisianw/FaceBoxes.PyTorch)，将他们放在这个目录下：

   ```bash
   $FaceBoxes_ROOT/data/WIDER_FACE/annotations/
   ```

   最终数据集目录结构如下：

   ![输入图片说明](https://images.gitee.com/uploads/images/2021/0927/121855_9a16b40b_6515416.png "屏幕截图.png")

2. 下载[FDDB](https://github.com/zisianw/FaceBoxes.PyTorch)数据集，将图片放在这个目录下（数据集包含2845张图片）：

   ```bash
   $FaceBoxes_ROOT/data/FDDB/images/
   ```

   最终数据集目录结构如下：

   ![输入图片说明](https://images.gitee.com/uploads/images/2021/0927/121924_9f00b12c_6515416.png "屏幕截图.png")

## Trainning

### 单卡性能评估

```bash
### 输出单卡FPS
bash scripts/train_performance_1p.sh
```

### 单卡训练

```bash
### 单卡全量训练
bash scripts/train_1p.sh
##  日志文件在当前目录下的1p_train.log
```

### 多卡性能评估

```bash
### 输出多卡FPS
bash scripts/train_performance_8p.sh
```

### 多卡训练

```bash
### 多卡全量训练
bash scripts/train_8p.sh
##  日志文件在当前目录下的8p_train.log
```

### Test

```bash
### 测试训练得到的权重文件，生成FDDB_dets.txt
bash test.sh
##  日志文件在当前目录下的test.log
### 解析FDDB_dets.txt文件，打印最终精度
bash eval.sh
```

## Performance

|         |   AP   | APEX | lOSS_SCALE | EPOCH |
| :-----: | :----: | :--: | :--------: | :---: |
| **GPU** | 0.9440 |  O2  |    128     |  300  |
| **NPU** | 0.9396 |  O2  |    128     |  300  |

