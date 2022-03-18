## PraNet

This implements training of PraNet on the Kvasir dataset, mainly modified from [PraNet](https://github.com/DengPingFan/PraNet)


## Requirements

1.Configuring your environment (Prerequisites):

* Install PyTorch 1.5.0 ([pytorch.org](http://pytorch.org))

* pip install -r requirements.txt

2.Downloading necessary data:

* Please refer to the method of PraNet implemented in [here](https://github.com/DengPingFan/PraNet).

## Training

To train a model, run `Train.py` with the desired model architecture and the path to the TrainDataset:

```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --train_path=./data/TrainDataset

# training 1p performance
bash ./test/train_performance_1p.sh --train_path=./data/TrainDataset

# training 8p accuracy
bash ./test/train_full_8p.sh --train_path=./data/TrainDataset

# training 8p performance
bash ./test/train_performance_8p.sh --train_path=./data/TrainDataset

# finetuning
bash test/train_finetune_1p.sh --train_path=./data/TrainDataset

# online inference demo 
python3.7.5 demo.py
```

Log path:
    test/output/devie_id/train_${device_id}.log           # training detail log
    test/output/devie_id/PraNet_bs16_8p_acc  # 8p training performance result log
    test/output/devie_id/train_PraNet_bs16_8p_acc_loss   # 8p training accuracy result log



## PraNet training result

| Acc@1 |  FPS   | Npu_nums | Epochs | AMP_Type |
| :---: | :----: | :------: | :----: | :------: |
|   -   | 9.216  |    1     |   1    |    O2    |
| 89.14 | 54.803 |    8     |   20   |    O2    |

