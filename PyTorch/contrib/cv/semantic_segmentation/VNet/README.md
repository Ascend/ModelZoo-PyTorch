# VNet

This implements training of VNet on the LUNA16 dataset, mainly modified from [mattmacy/vnet.pytorch](https://github.com/mattmacy/vnet.pytorch).

## VNet Detail

VNet is trained end-to-end on MRI volumes depicting prostate, and learns to predict segmentation for the whole volume at once.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
  (SimpleITK on ARM needs to be installed by source code)
- Download the LUNA16 dataset from https://luna16.grand-challenge.org/Download/
    - Then use normalize_dataset.py to get the preprocessed dataset.

## Training

To train a model, run `train.py` with the desired model architecture and the path to the LUNA16 dataset:

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
    test/output/devie_id/VNet_bs4_8p_perf.log  # 8p training performance result log
    test/output/devie_id/VNet_bs4_8p_acc.log   # 8p training accuracy result log



## VNet training result

| Error rate  | FPS    | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| 0.414%   | 17.93     | 1        | 200        | O2       |
| 0.745%   | 123.16    | 8        | 200      | O2       |

