# RegNetY-1.6GF

This implements training of RegNetY-1.6GF on the ImageNet dataset, mainly modified from [GitHub](https://github.com/facebookresearch/pycls).

## RegNetY-1.6GF Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, RegNetY-1.6GF is re-implemented using semantics such as custom OP.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
  Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

To train a model, run `imagenet_fast.py` with the desired model architecture and the path to the ImageNet dataset:

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


## RegNetY-1.6GF training result

| Acc@1    | FPS       | Npu_nums | Epochs   | Initial LR | AMP_Type |
| :------: | :------:  | :------: | :------: | :--------: | :------: |
| -        | 892       | 1        | 1        | 0.0125     | O2       |
| 77.053   | 4445      | 8        | 90       | 0.1        | O2       |