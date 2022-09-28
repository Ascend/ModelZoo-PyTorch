# Convmixer: Patches Are All You Need?
https://github.com/locuslab/convmixer

## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org)), torch >= 1.4.0, torchvision >= 0.5.0
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/

## 软件包
- 910版本
- CANN_toolkit_5.1.RC2.alpha001
- torch 1.5.0+ascend.post5.20220505

## Training the convmixer

To train a model, run `train_npu.py` with the desired model architecture and the path to the ImageNet dataset:

training on 8 npu
```
bash ./test/train_full_8p.sh  --data_path=xxx       

```

get model performance
```
1. test 1p performance  
bash ./test/train_performance_1p.sh  --data_path=xxx   

2. test 8p performance  
bash ./test/train_performance_8p.sh  --data_path=xxx   
```

## Validation
```
bash ./test/train_eval_1p.sh  --data_path=xxx --checkpoint=XXX
```

## Traing log
```
test/output/${device_id}/train_${device_id}.log              # training detail log

test/output/${device_id}/${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'.log            # training accuracy result

```

## Convmixer  training result
|  name   | Acc@1   |   FPS   |   Epochs   | AMP_Type |
|:-------:| :-----: |:-------:|:----------:| :------: |
| NPU-1p  |   -     |  50.24  |    150     |    O2    |
| NPU-8p  | 80.2%   | 407.18  |    150     |    O2    |

