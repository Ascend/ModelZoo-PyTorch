# MobileNetV3_large_100_for_PyTorch

This implements training mobilenetv3_large_100 of  on the ImageNet dataset, mainly modified from https://github.com/rwightman/pytorch-image-models  

## MobileNetV3_large_100_for_PyTorch Detail 

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 



## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- pip install timm
- pip install git+https://github.com/rwightman/pytorch-image-models.git
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
## 软件包
- 910版本
- CANN toolkit_5.1.RC1.alpha001
- torch 1.8.1.rc2.20220505
- 固件驱动 22.0.0

## Training 

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path

#  eval 
bash test/train_eval.sh

#To ONNX
python3 pthtar2onx.py  --model-path path/to/model_best.pth.tar 

# online inference demo 
python3 demo.py --model-path /path/to/model_best.pth.tar

```

## MobileNetV3_large_100_for_PyTorch training result 

|  Acc@1  |   FPS    | Npu_nums | Epochs | AMP_Type |
| :-----: | :------: | :------: | :----: | :------: |
|    -    |    616      |    1     |  3   |    O2    |
| 74.9339 | 9638 |    8     |  600   |    O2    |



