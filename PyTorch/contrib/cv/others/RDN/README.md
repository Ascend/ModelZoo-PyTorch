# RDN 训练
# Residual Dense Network for Image Super-Resolution
This implements training of RDN on the DIV2K_x2 dataset.
- Reference implementation：
```
url=https://github.com/yjn870/RDN-pytorch
```

## RDN Detail # 

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore, RDN is re-implemented using semantics such as custom OP. 


## Requirements # 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- The DIV2k, Set5 Dataset can be downloaded from the links below.Move the datasets to directory ./data .
    - Train Set : [Download DIV2k](https://www.dropbox.com/s/41sn4eie37hp6rh/DIV2K_x2.h5?dl=0)
    - Test Set : [Download Set5](https://www.dropbox.com/s/pd52pkmaik1ri0h/rdn_x2.pth?dl=0)

## Training # 
To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
# 1p train perf
bash test/train_performance_1p.sh

# 8p train perf
bash test/train_performance_8p.sh

# 8p train full
bash test/train_full_8p.sh

# 8p eval
bash test/train_eval_8p.sh

# finetuning
bash test/train_finetune_1p.sh
```

## RDN training result # 

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 240      | 1        | 800      | O1       |
| 37.95     | 1716     | 8        | 800      | O1       |



