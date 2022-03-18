# MGN

This implements training of MGN on the Market-1501 dataset, mainly modified from [GNAYUOHZ/ReID-MGN](https://github.com/GNAYUOHZ/ReID-MGN).

## MGN Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, MGN is re-implemented using semantics such as custom OP.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))  

- `pip install -r requirements.txt`

- Download the Market-1501 dataset from https://paperswithcode.com/dataset/market-1501

  - ~~~shell
    unzip Market-1501-v15.09.15.zip
    ~~~

## Training

To train a model, run `main.py` with the desired model architecture and the path to the market dataset:

```bash
# training 1p accuracy
bash test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash test/train_performance_8p.sh --data_path=real_data_path

# finetune
bash test/train_finetune_1p.sh --data_path=real_data_path --weights=real_weight_path

# Online inference demo
python demo.py --data_path real_data_path

# To ONNX
python pthtar2onnx.py 
```

## MGN training result


|        |  mAP  | AMP_Type | Epochs |   FPS   |
| :----: | :---: | :------: | :----: | :-----: |
| 1p-GPU |   -   |    O2    |   1    | 71.408  |
| 1p-NPU |   -   |    O2    |   1    | 29.408  |
| 8p-GPU | 93.35 |    O2    |  500   | 771.818 |
| 8p-NPU | 93.83 |    O2    |  500   | 200.024 |

