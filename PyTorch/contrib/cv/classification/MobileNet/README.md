# MobileNet

This implements training of MobileNet on the ImageNet dataset, mainly modified from [Github](https://github.com/pytorch/examples/tree/master/imagenet). 

## MobileNet Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. Therefore, MobileNet is re-implemented using semantics such as custom OP. For details, see mobilenet.py.

## Requirements

- pytorch_ascend, apex_ascend, tochvision
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
  Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
## Training

To train a model, run `mobilenet.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
# 1p train 
bash ./test/train_full_1p.sh  --data_path=数据集路径    # 精度训练
bash ./test/train_performance_1p.sh  --data_path=数据集路径  # 性能训练

# 8p train
bash ./test/train_full_8p.sh  --data_path=数据集路径         # 精度训练
bash ./test/train_performance_8p.sh  --data_path=数据集路径  # 性能训练

# eval default 8p， should support 1p
`bash ./test/train_eval_8p.sh  --data_path=数据集路径`

# online inference demo
bash test/run_demo.sh

# To ONNX
bash test/run_2onnx.sh
```

## MobileNet training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 1574.13      | 1        | 1      | O2       |
| 70.972     | 10912.79     | 8        | 90      | O2       |


