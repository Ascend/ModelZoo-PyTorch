# SRCNN

This repository is implementation of the ["Image Super-Resolution Using Deep Convolutional Networks"](https://arxiv.org/abs/1501.00092)(SRCNN)by PyTorch.

## 参考实现
<https://github.com/fuyongXu/SRCNN_Pytorch_1.0/blob/master/train.py>

## Requirements

- PyTorch 1.4
- Numpy 1.17.0
- Pillow 6.2.0
- h5py 3.6.0
- tqdm 4.64.0

## Data
Use the 91-image dataset as the training set and the Set-5 dataset as the validation set.

## Train
```python
python3.7 train.py --data_url='data_path' --train_url='output_path'
```

## result
|名称|精度（PSNR）|
|----|----|
|paper|32.75|
|GPU|33.27|
|NPU|33.27|


