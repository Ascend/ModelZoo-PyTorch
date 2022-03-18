# ResNeXt-101-32x8d

This implements training of ResNeXt-101-32x8d on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

## ResNeXt-101-32x8d Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, ResNeXt-101-32x8d is re-implemented using semantics such as custom OP.


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

# O2 online inference demo
source scripts/set_npu_env.sh
python3.7.5 demo.py

# O2 To ONNX
source scripts/set_npu_env.sh
python3.7.5 pthtar2onnx.py


```

## ResNeXt-101-32x8d training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 345       | 1        | 1        | O2       |
| 79.173   | 2673      | 8        | 90       | O2       |
