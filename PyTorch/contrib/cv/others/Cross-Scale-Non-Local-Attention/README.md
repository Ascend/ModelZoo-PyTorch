# CSNLA

This is for CSNLA. The code is built on [EDSR(PyTorch)](https://github.com/sanghyun-son/EDSR-PyTorch).


## CSNLA Detail

Details, see src/model/csnln.py


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the DIV2K and Set5 datasets, and pretrained_models by referring to [Cross-Scale-Non-Local-Attention](https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention)

## Training and Testing

```
# switch to the dir src
cd src
```


To train a model, run 'main.py' with the desired model architecture and the path of the DIV2K dataset. To test a trained model, run 'main.py' with pretained model and the path of the Set5 dataset.

```bash
# xxx is the decompressed directory of datasets.zip, such as /home/CSNLA
# 1p train perf
bash ../test/train_performance_1p.sh --data_path=xxx

# 8p train perf
bash ../test/train_performance_8p.sh --data_path=xxx

# 8p train full
# Remarks: Target accuracy 37.12; test accuracy 36.969
bash ../test/train_full_8p.sh --data_path=xxx 
```


## CSNLA training result
|  名称  | 精度  | 性能 | AMP_Type |
| :----: | ----- | ---- | -------- |
| NPU-1p | -     | 0.67  | O2       |
| NPU-8p | 36.979 | 4.5 | O2       |


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md