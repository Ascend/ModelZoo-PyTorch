# MobileNetV3
An implementation of MobileNetV3 with pyTorch

# Theory
&emsp;You can find the paper of MobileNetV3 at [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244).

# Prepare data

* CIFAR-10
* CIFAR-100
* SVHN
* Tiny-ImageNet
* ImageNet: Please move validation images to labeled subfolders, you can use the script [here](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

# Train

* Train from scratch:

```
CUDA_VISIBLE_DEVICES=3 python train.py --batch-size=128 --mode=small \
--print-freq=100 --dataset=CIFAR100 --ema-decay=0 --label-smoothing=0.1 \
--lr=0.3 --save-epoch-freq=1000 --lr-decay=cos --lr-min=0 \
--warmup-epochs=5 --weight-decay=6e-5 --num-epochs=200 --width-multiplier=1 \
-nbd -zero-gamma -mixup
```

where the meaning of the parameters are as followed:

```
batch-size
mode: using MobileNetV3-Small(if set to small) or MobileNetV3-Large(if set to large).
dataset: which dataset to use(CIFAR10, CIFAR100, SVHN, TinyImageNet or ImageNet).
ema-decay: decay of EMA, if set to 0, do not use EMA.
label-smoothing: $epsilon$ using in label smoothing, if set to 0, do not use label smoothing.
lr-decay: learning rate decay schedule, step or cos.
lr-min: min lr in cos lr decay.
warmup-epochs: warmup epochs using in cos lr deacy.
num-epochs: total training epochs.
nbd: no bias decay.
zero-gamma: zero $gamma$ of last BN in each block.
mixup: using Mixup.
```

# Pretrained models

&emsp;We have provided the pretrained MobileNetV3-Small model in `pretrained`.

# Experiments

## Training setting

### on ImageNet

```
CUDA_VISIBLE_DEVICES=5 python train.py --batch-size=128 --mode=small --print-freq=2000 --dataset=imagenet \
--ema-decay=0.99 --label-smoothing=0.1 --lr=0.1 --save-epoch-freq=50 --lr-decay=cos --lr-min=0 --warmup-epochs=5 \
--weight-decay=1e-5 --num-epochs=250 --num-workers=2 --width-multiplier=1 -dali -nbd -mixup -zero-gamma -save
```

### on CIFAR-10

```
CUDA_VISIBLE_DEVICES=1 python train.py --batch-size=128 --mode=small --print-freq=100 --dataset=CIFAR10\
  --ema-decay=0 --label-smoothing=0 --lr=0.35 --save-epoch-freq=1000 --lr-decay=cos --lr-min=0\
  --warmup-epochs=5 --weight-decay=6e-5 --num-epochs=400 --num-workers=2 --width-multiplier=1
```

### on CIFAR-100

```
CUDA_VISIBLE_DEVICES=1 python train.py --batch-size=128 --mode=small --print-freq=100 --dataset=CIFAR100\
  --ema-decay=0 --label-smoothing=0 --lr=0.35 --save-epoch-freq=1000 --lr-decay=cos --lr-min=0\
  --warmup-epochs=5 --weight-decay=6e-5 --num-epochs=400 --num-workers=2 --width-multiplier=1
```

&emsp;Using more tricks：
```
CUDA_VISIBLE_DEVICES=1 python train.py --batch-size=128 --mode=small --print-freq=100 --dataset=CIFAR100\
  --ema-decay=0.999 --label-smoothing=0.1 --lr=0.35 --save-epoch-freq=1000 --lr-decay=cos --lr-min=0\
  --warmup-epochs=5 --weight-decay=6e-5 --num-epochs=400 --num-workers=2 --width-multiplier=1\
  -zero-gamma -nbd -mixup
```

### on SVHN

```
CUDA_VISIBLE_DEVICES=3 python train.py --batch-size=128 --mode=small --print-freq=1000 --dataset=SVHN\
  --ema-decay=0 --label-smoothing=0 --lr=0.35 --save-epoch-freq=1000 --lr-decay=cos --lr-min=0\
  --warmup-epochs=5 --weight-decay=6e-5 --num-epochs=20 --num-workers=2 --width-multiplier=1
```

### on Tiny-ImageNet

```
CUDA_VISIBLE_DEVICES=7 python train.py --batch-size=128 --mode=small --print-freq=100 --dataset=tinyimagenet\
  --data-dir=/media/data2/chenjiarong/ImageData/tiny-imagenet --ema-decay=0 --label-smoothing=0 --lr=0.15\
  --save-epoch-freq=1000 --lr-decay=cos --lr-min=0 --warmup-epochs=5 --weight-decay=6e-5 --num-epochs=200\
  --num-workers=2 --width-multiplier=1 -dali
```

&emsp;Using more tricks：
```
CUDA_VISIBLE_DEVICES=7 python train.py --batch-size=128 --mode=small --print-freq=100 --dataset=tinyimagenet\
  --data-dir=/media/data2/chenjiarong/ImageData/tiny-imagenet --ema-decay=0.999 --label-smoothing=0.1 --lr=0.15\
  --save-epoch-freq=1000 --lr-decay=cos --lr-min=0 --warmup-epochs=5 --weight-decay=6e-5 --num-epochs=200\
  --num-workers=2 --width-multiplier=1 -dali -nbd -mixup
```

## MobileNetV3-Large

### on ImageNet

|              | Madds     | Parameters | Top1-acc  | Top5-acc  |
| -----------  | --------- | ---------- | --------- | --------- |
| Offical 1.0  | 219 M     | 5.4  M     | 75.2%     |     -     |
| Ours    1.0  | 216.6 M   | 5.47 M     | -         |     -     |

### on CIFAR-10

|              | Madds     | Parameters | Top1-acc  | Top5-acc  |
| -----------  | --------- | ---------- | --------- | --------- |
| Ours    1.0  | 66.47 M   | 4.21 M     | -         |     -     |

### on CIFAR-100

|              | Madds     | Parameters | Top1-acc  | Top5-acc  |
| -----------  | --------- | ---------- | --------- | --------- |
| Ours    1.0  | 66.58 M   | 4.32 M     | -         |     -     |

## MobileNetV3-Small

### on ImageNet

|              | Madds     | Parameters | Top1-acc  | Top5-acc  |
| -----------  | --------- | ---------- | --------- | --------- |
| Offical 1.0  | 56.5 M    | 2.53 M     | 67.4%     |     -     |
| Ours    1.0  | 56.51 M   | 2.53 M     | 67.52%    | 87.58%    |

&emsp;The pretrained model with top-1 accuracy 67.52% is provided in the folder [pretrained](https://github.com/ShowLo/MobileNetV3/tree/master/pretrained).

### on CIFAR-10 (Average accuracy of 5 runs)

|              | Madds     | Parameters | Top1-acc  | Top5-acc  |
| -----------  | --------- | ---------- | --------- | --------- |
| Ours    1.0  |  17.51 M  |   1.52 M   |   92.97%  |     -     |

### on CIFAR-100 (Average accuracy of 5 runs)

|              | Madds     | Parameters | Top1-acc  | Top5-acc  |
| -----------  | --------- | ---------- | --------- | --------- |
| Ours    1.0  |  17.60 M  |   1.61 M   |  73.69%   |  92.31%   |
| More Tricks  |   same    |    same    |  76.24%   |  92.58%   |

### on SVHN (Average accuracy of 5 runs)

|              | Madds     | Parameters | Top1-acc  | Top5-acc  |
| -----------  | --------- | ---------- | --------- | --------- |
| Ours    1.0  |  17.51 M  |   1.52 M   |   97.92%  |     -     |

### on Tiny-ImageNet (Average accuracy of 5 runs)

|              | Madds     | Parameters | Top1-acc  | Top5-acc  |
| -----------  | --------- | ---------- | --------- | --------- |
| Ours    1.0  |  51.63 M  |   1.71 M   |  59.32%   |  81.38%   |
| More Tricks  |   same    |    same    |  62.62%   |  84.04%   |

## Dependency

&emsp;This project uses Python 3.7 and PyTorch 1.1.0. The FLOPs and Parameters and measured using [torchsummaryX](https://github.com/nmhkahn/torchsummaryX).
