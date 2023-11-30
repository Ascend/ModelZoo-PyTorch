# GAN 训练
This implements training of RDN on the DIV2K_x2 dataset.
- Reference implementation：
```
url=https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
```



## Requirements # 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- The MNIST Dataset can be downloaded from the links below.
    - Train Set : [Download Mnist](https://wwr.lanzoui.com/iSBOeu43dkf)

## Training # 
To train a model, run: 

```bash
# 1p train perf
bash train_performance_1p.sh --data_path=data/mnist

# 8p train perf
bash train_performance_8p.sh --data_path=data/mnist

# 8p train full
bash train_full_8p.sh --data_path=data/mnist

# 8p eval
bash train_eval_8p.sh --data_path=data/mnist

# finetuning
bash train_finetune_1p.sh --data_path=data/mnist
```
After running,you can see the results in `./output`

## GAN training result # 

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 1642.130      | 1        | 200      | O1       |
| -     | 15275.049     | 8        | 200      | O1       |


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md
