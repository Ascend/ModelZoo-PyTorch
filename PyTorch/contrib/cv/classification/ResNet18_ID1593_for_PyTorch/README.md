# ResNet18_ID1593_for_PyTorch 

This implements training of ResNet18_ID1593_for_PyTorch  on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).



## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Suggesting the version of Pillow is 9.1.0 and the version of torchvision is 0.6.0.Torchvision can be installed from its source code https://github.com/pytorch/vision
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

# eval default 8p， should support 1p
bash ./test/train_eval_8p.sh  --data_path=xxx

## Traing log
test/output/devie_id/train_${device_id}.log              # training detail log

test/output/devie_id/ResNet18_${device_id}_bs_8p_perf.log            # 8p training performance result

test/output/devie_id/ResNet18_${device_id}_bs_8p_acc.log             # 8p training accuracy result

## ResNet18_ID1593_for_PyTorch  training result
| Acc@1 | FPS  | Npu_nums | Epochs | AMP_Type |
| :---: | :--: | :------: | :----: | :------: |
|   -   | 4121 |    1     |  120   |    O2    |
| 69.91 | 6400 |    8     |  120   |    O2    |

