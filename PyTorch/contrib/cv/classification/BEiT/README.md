# BEiT: BERT Pre-Training of Image Transformers

This implements training of BEiT on the ImageNet dataset, mainly modified from https://github.com/microsoft/unilm/tree/master/beit


## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/

## Training 
- To train a model, run `run_class_finetuning_apex_npu.py` with the desired model architecture and the path to the ImageNet dataset:

- Download the `beit_base_patch16_224_pt22k_ft22k.pth` to `./checkpoints` from
https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k_ft22k.pth



```
# 1p training 1p
bash ./test/train_full_1p.sh  --data_path=xxx          # training accuracy

bash ./test/train_performance_1p.sh  --data_path=xxx   # training performance

# 8p training 8p
bash ./test/train_full_8p.sh  --data_path=xxx          # training accuracy

bash ./test/train_performance_8p.sh  --data_path=xxx   # training performance

# eval default 1p， should support 8p
bash ./test/train_eval_1p.sh  --data_path=xxx --resume=XXX

```

## BEit  training result
| name | Acc@1 | FPS  | Npu_nums | Epochs | AMP_Type | s/ per step  |
| :--: | :---: | :--: | :------: | :----: | :------: | :---------:  |
| NPU  |   -   | 162  |     1    |   30   |    O2    |    0.414     |
| NPU  |85.279 | 1210 |     8    |   30   |    O2    |    0.422     |

FPS = BatchSize * num_devices / time_avg