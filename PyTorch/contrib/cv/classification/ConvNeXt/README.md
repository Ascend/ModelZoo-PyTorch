# ConvNext_for_PyTorch

This implements training ConvNext of  on the ImageNet dataset, mainly modified from https://github.com/facebookresearch/ConvNeXt.git 

## ConvNext_for_PyTorch Detail 

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 



## Requirements 
- pip install -r requirements.txt
- pip install torch==1.8.1+ascend.rc2.20220505;torchvision==0.9.1;torch-npu 1.8.1rc2.post20220505;
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
## timm 
将timm_need目录下的文件替换到timm的安装目录下
```bash

cd ../ConvNeXt
/bin/cp -f timm_need/mixup.py ../timm/data/mixup.py
/bin/cp -f timm_need/model_ema.py ../timm/utils/model_ema.py

```
## 软件包
- 910版本
- CANN toolkit_5.1.RC1
- torch 1.8.1+ascend.rc2.20220505
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
bash test/train_eval_8p.sh --data_path=real_data_path

# finetuning
bash test/train_finetune_1p.sh --data_path=real_data_path


```

## ConvNext_for_PyTorch training result 

|  Acc@1  |   FPS    | Npu_nums | Epochs | AMP_Type |
| :-----: | :------: | :------: | :----: | :------: |
|    -    |    115.10      |    1     |  300   |    O1    |
| 82.049 | 259.85 |    8     |  300   |    O1    |


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md




