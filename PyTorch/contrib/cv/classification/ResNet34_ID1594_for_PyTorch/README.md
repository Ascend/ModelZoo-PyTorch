# ResNet34_ID1594_for_PyTorch

This implements training of ResNet34_ID1594_for_PyTorch on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet). # 简要说明这个repo做了什么，迁移自哪里，参考了哪里

## ResNet34_ID1594_for_PyTorch Detail 



## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
  Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training 

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
#全量精度执行
bash ./test/train_full_8p.sh --data_path=/npu/traindata/imagenet_pytorch

#性能执行命令
bash ./test/train_performance_8p.sh --data_path=/npu/traindata/imagenet_pytorch
```

## ResNet34_ID1594_for_PyTorch training result # 性能展示-必写
| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 1906       | 1        | 130      | O2       |
| 73.3     | 6266     | 8        | 130      | O2       |

# Else # 其他说明，请自行补充-可选

