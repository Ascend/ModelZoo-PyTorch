# pytorch-GENet

An unofficial Pytorch implementation of https://arxiv.org/abs/1810.12348. Probably.

The code replaces the standard blocks in a WideResNet with GEBlocks and trains these models on CIFAR-10/100. The blocks are defined in `models/blocks.py`

The code is currently untested, so ... see what happens when you run it.

## Setup
Clean conda env as usual.

```
conda create -n prunes python=3.6
conda activate prunes
conda install pytorch torchvision -c pytorch
```

## Running

All the various GE plus, minus, standards can be used by changing the following input arguments:

-`extent`: The extent factor. Set to 0 for global

-`extra_params`: Whether there are learnable parameters for downsampling

-`mlp`: Whether to use a squeeze-excite style MLP after downsampling  


e.g. to train a WRN-16-8 with GE theta-minus blocks and global extent use:
```
python train.py --depth 16 --width 8 --extent 0  --extra_params False --mlp False
```
To train a WRN-16-8 with GE theta blocks and global extent, use:
```
python train.py --depth 16 --width 8 --extent 0  --extra_params True --mlp False
```
To train a WRN-16-8 with GE theta-plus blocks and extent 2, use:
```
python train.py --depth 16 --width 8 --extent 2  --extra_params True --mlp True
```
 
and so on, and so forth.

### Acknowledgements

Base code for Wideresnet training was borrowed from
 ```https://github.com/xternalz/WideResNet-pytorch```
 
And thanks to the authors of the actual paper. 
