# Resnext-101-32x8d

This implements training of Resnext-101-32x8d on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet). 

## Resnext-101-32x4d Detail 

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore, Resnext-101-32x8d is re-implemented using semantics such as custom OP. For details, see models/main.py . 


## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training 

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

# 1p training 1p
bash ./test/train_full_1p.sh  --data_path=xxx          # training accuracy

bash ./test/train_performance_1p.sh  --data_path=xxx   # training performance

# 8p training 8p
bash ./test/train_full_8p.sh  --data_path=xxx          # training accuracy

bash ./test/train_performance_8p.sh  --data_path=xxx   # training performance

## Traing log
test/output/devie_id/train_${device_id}.log                                  # training detail log

test/output/devie_id/Resnext-101-32x8d_${device_id}_bs_8p_perf.log            # 8p training performance result

test/output/devie_id/Resnext-101-32x8d_${device_id}_bs_8p_acc.log             # 8p training accuracy result

## Resnext-101-32x8d training result 

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        |    436    | 1        |  1       | O2       |
| 78.884   |   2207    | 8        |  90      | O2       |

# Else 
