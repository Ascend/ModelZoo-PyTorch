# VGG-16

This implements training of vgg19 on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

## VGG-16 Detail

As of the current date, Ascend-Pytorch is still have some bug in nn.Dropout(). For details, see ./vgg.py


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

# eval default 8p， should support 1p
bash ./test/train_eval_8p.sh  --data_path=xxx

## Traing log
test/output/devie_id/train_${device_id}.log                       # training detail log

test/output/devie_id/Vgg16_${device_id}_bs_8p_perf.log            # 8p training performance result

test/output/devie_id/Vgg16_${device_id}_bs_8p_acc.log             # 8p training accuracy result

## VGG-16 training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 360       | 1        | 240      | O2       |
| 72.933   | 2500      | 8        | 240      | O2       |
