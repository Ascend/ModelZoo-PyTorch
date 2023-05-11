# RegNetX

This implements training of RegNetX on the ImageNet dataset, mainly modified from [facebookresearch/pycls](https://github.com/facebookresearch/pycls).

## RegNetX Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, RegNetX is re-implemented using semantics such as custom OP.


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

# 8p eval
bash test/train_eval_8p.sh --data_path=real_data_path --pth_path=./checkpoints/model_best.pth.tar

# finetuning
bash test/train_finetune_1p.sh --data_path=real_data_path --pth_path=./checkpoints/checkpoint.pth.tar

# online inference demo 
python3 demo.py
```

Log path:
    test/output/devie_id/train_${device_id}.log           # training detail log
    test/output/devie_id/RegNetX_8p_perf.log  # 8p training performance result log
    test/output/devie_id/RegNetX_8p_acc.log   # 8p training accuracy result log



## RegNetX training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 117       | 1        | 1        | O2       |
| 77.167   | 7460      | 8        | 100      | O2       |
