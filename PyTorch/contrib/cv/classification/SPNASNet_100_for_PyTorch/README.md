
This implements training of  spnasnet_100 ImageNet dataset, mainly modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py

## Inception V4 Detail 

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
For details, see https://github.com/rwightman/pytorch-image-models


## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- git clone https://github.com/Cadene/pretrained-models.pytorch.git
- Download the ImageNet dataset from http://www.image-net.org/
  - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
--pip3.7 install -r requirements.txt

##Training

- training 1p 
    - bash ./test/train_full_1p.sh --data_path=xxx # training accuracy

    - bash ./test/train_performance_1p.sh --data_path=xxx # training performance

- training 8p
    - bash ./test/train_full_8p.sh --data_path=xxx # training accuracy
    
    - bash ./test/train_performance_8p.sh --data_path=xxx # training performance

- eval default 8p， should support 1p
    - bash ./test/train_eval_8p.sh --data_path=xxx

- Traing log
    - test/output/devie_id/train_${device_id}.log # training detail log
    
    - test/output/devie_id/ResNet101_${device_id}_bs_8p_perf.log # 8p training performance result
    
    - test/output/devie_id/ResNet101_${device_id}_bs_8p_acc.log # 8p training accuracy result    

```bash

# online inference demo 
python3.7 demo.py

# pth转换onnx
python3.7 pthtar2onnx.py

```

## Inception V4 training result 

|  Acc@1  |   FPS    | Npu_nums | Epochs | AMP_Type |
| :-----: | :------: | :------: | :----: | :------: |
|    -    |  387.452 |    1     |  1     |    O1    |
| 74.571  | 2988.835 |    8     |  150   |    O1    |
