## Abstract

This implements training of ShuffleNetV2 on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

## Details

Base version of the model from [shufflenetv2.py](https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py)
As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore, ShufflenetV2 is re-implemented using semantics such as custom OP. For details, see models/shufflenetv2_wock_op_woct.py .


## Requirements

- Ascend_Pytorch, Ascend_Apex, tochvision
- Download the dataset from http://www.image-net.org/
  - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)


## Installing Pillow from source

It is recommended to install pillow 9.1.0 from source following the steps below:

```
git clone https://github.com/python-pillow/Pillow.git
cd Pillow 
git checkout 9.1.0 
python3.7 setup.py install
```

If you are prompted that some dependencies are missing, please install them following https://pillow.readthedocs.io/en/stable/installation.html

## Training 

### 1p training

```
# Please checkout data path,for example: /data/imagenet/ 
 bash ./test/train_full_1p.sh  --data_path=/data/imagenet/
```

### 8p training

```
# Please checkout data path,for example: /data/imagenet/ 
bash ./test/train_full_8p.sh  --data_path=/data/imagenet/
```

### Docker training

1. Import docker image

```
docker import ubuntuarmpytorch.tar pytorch:b020
```

2. Run docker_start.sh with corresponding parameters

```
./docker_start.sh pytorch:b020 /train/imagenet /home/Shufflenetv2_for_PyTorch
```

3. Perform step 1 training process

## Result

The save path is shown below：

```
/home/Shufflenetv2_for_Pytorch/result/training_8p_job_20201121023601
```

### Training result

| Acc@1 |  FPS  | Npu_nums | Epochs | Type |
| :---: | :---: | :------- | :----: | :--: |
| 61.5  | 1200  | 1        |   20   |  O2  |
| 68.5  | 2200  | 1        |  240   |  O2  |
| 66.3  | 14000 | 8        |  240   |  O2  |

- The 8p training precision is the same as that of the GPU. Compared with the 1p training precision, this is mainly caused by the large batch size (8192). You can use a distributed optimizer such as LARS to resolve this problem.

## Note

Some numpy versions will cause errors during operation. Please avoid them: 1.19.2