# VoVNet-39

This implements training of VoVNet-39 on the ImageNet dataset, mainly modified from [GitHub](https://github.com/paynezhangpayne/vovnet-detectron2).

## VoVNet-39 Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, VoVNet-39 is re-implemented using semantics such as custom OP.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- `torchvision==0.5.0(x86) && torchvision==0.2.0(arm)`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path
```

Log path:
    - test/output/devie_id/train_${device_id}.log                  # training detail log
    - test/output/devie_id/VoVNet39_for_PyTorch_bs128_8p_perf.log  # 8p training performance result log
    - test/output/devie_id/VoVNet39_for_PyTorch_bs128_8p_acc.log   # 8p training accuracy result log


## Running demo

Run `demo.py` as the demo script with trained model (`model.pth` as example):

```bash
python3 demo.py model.pth
```


## VoVNet-39 training result

| Acc@1    | FPS       | Npu_nums | Epochs   | Initial LR | AMP_Type |
| :------: | :------:  | :------: | :------: | :--------: | :------: |
| -        | 892       | 1        | 1        | 0.0125     | O2       |
| 77.053   | 4445      | 8        | 90       | 0.1        | O2       |


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md