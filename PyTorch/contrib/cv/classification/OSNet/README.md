# OSNet

This implements training of OSNet on the Market-1501 dataset, mainly modified from [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid).

## OSNet Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, OSNet is re-implemented using semantics such as custom OP.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))  


- `pip install -r requirements.txt`

- Install torchreid

  - ~~~python
    python setup.py develop
    ~~~

- Download the Market-1501 dataset from https://paperswithcode.com/dataset/market-1501

  - ~~~shell
    unzip Market-1501-v15.09.15.zip
    ~~~
  
- Move Market-1501 dataset to 'reid-data' path

  - ~~~shell
    mkdir path_to_osnet/reid-data/
    mv Market-1501-v15.09.15 path_to_osnet/reid-data/market1501 
    ~~~
## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
# training 1p accuracy
bash test/train_full_1p.sh

# training 1p performance
bash test/train_performance_1p.sh

# training 8p accuracy
bash test/train_full_8p.sh

# training 8p performance
bash test/train_performance_8p.sh

# finetuning
bash test/train_finetune_1p.sh --data_path=real_data_path --weight=real_weight_path

# Online inference demo
python demo.py
## 备注： 识别前后图片保存到 `inference/` 文件夹下

# To ONNX
python pthtar2onnx.py 
```

## OSNet training result


|        | mAP  | AMP_Type | Epochs |   FPS    |
| :----: | :--: | :------: | :----: | :------: |
| 1p-GPU |  -   |    O2    |   1    | 371.383  |
| 1p-NPU |  -   |    O2    |   1    | 366.464  |
| 8p-GPU | 80.3 |    O2    |  350   | 1045.535 |
| 8p-NPU | 80.2 |    O2    |  350   | 1091.358 |


# 公网地址说明

代码涉及公网地址参考 public_address_statement.md
