# FFDNet_pytorch
+ A PyTorch implementation of a denoising network called [FFDNet](https://github.com/cszn/FFDNet)
+ Paper: FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising - [arxiv](https://arxiv.org/abs/1710.04026) / [IEEE](https://ieeexplore.ieee.org/abstract/document/8365806/)

### Dataset

+ [Waterloo Exploration Database](https://ece.uwaterloo.ca/~k29ma/exploration/)

### Usage

+ Train

```bash
python3 ffdnet.py \
    --use_gpu \
    --is_train \
    --train_path './train_data/' \
    --model_path './models/' \
    --batch_size 768 \
    --epoches 80 \
    --val_epoch 5
    --patch_size 32 \
    --save_checkpoints 20 \
    --train_noise_interval 15 75 15 \
    --val_noise_interval 30 60 30 \
```

+ Test

```bash
python3 ffdnet.py \
    --use_gpu \
    --is_test \
    --test_path './test_data/color.png' \
    --model_path './models/' \
    --add_noise
    --noise_sigma 30
```

### References

+ Some codes are copied from [An Analysis and Implementation of the FFDNet Image Denoising Method](http://www.ipol.im/pub/pre/231/)