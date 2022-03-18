# WideResnet50_2

This implements training of WideResnet50_2 on the ImageNet dataset, mainly modified from [pytorch/examples](https://github.com/pytorch/examples/tree/master/imagenet).

## WideResnet50_2 Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, WideResnet50_2 is re-implemented using semantics such as custom OP.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

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
```

Log path:
    test/output/devie_id/train_${device_id}.log           # training detail log
    test/output/devie_id/WideReesnet50_2_bs8192_8p_perf.log  # 8p training performance result log
    test/output/devie_id/WideReesnet50_2_bs8192_8p_acc.log   # 8p training accuracy result log



## WideResnet50_2 training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 646       | 1        | 1        | O2       |
| 78.756   | 2500      | 8        | 200      | O2       |

