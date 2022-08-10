# Twins_PCPVT训练指导

Twins_PCPVT模型在imagenet数据集上进行训练， 并将其迁移到NPU，代码主要修改自[官方仓库](https://github.com/Meituan-AutoML/Twins).

## 迁移过程

1. 迁移到 NPU 上
2. 支持分布式训练和数据并行
3. 使用apex进行混合精度训练

## 训练细节

- 目前，Ascend-Pytorch 不支持timm库，因为涉及到对timm库cuda部分的修改，因此将timm源码也加入了仓库中。
- 目前，NPU不支持torch1.7版本，但是代码是1.7的，因此用apex库替换了torch原生库。
- 目前，使用apex混合精度和NPU fused优化器后，性能大幅提升但是精度略有下降，但在可接受的范围内。

## 准备 Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## 训练 Training

训练和性能验证脚本如下，首先进入脚本目录，`cd test/`

```bash
# training 1p performance
nohup bash train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
nohup bash train_full_8p.sh --data_path=real_data_path

# training 8p performance
nohup bash train_performance_8p.sh --data_path=real_data_path

# finetuning 1p
nohup bash train_finetune_1p.sh --data_path=real_data_path --finetune_pth=real_checkpoint_path

```

## Twins PVPVT 训练结果

| 实验设备 |    FPS     | Epoch | Acc@1  |
| :------: |:----------:|:-----:|:------:|
|  8p-GPU  |  1760.448  |  100  | 73.87% |
|  8p-NPU  |  1940.732  |  100  | 72.79% |
|  1p-NPU  |   292.89   |   -   |   -    |

