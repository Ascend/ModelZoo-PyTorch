# GAN 训练
This implements training of RDN on the DIV2K_x2 dataset.
- Reference implementation：
```
url=https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
```



## Requirements # 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- The MNIST Dataset can be downloaded from the links below.Move the datasets to directory ./data .
    - Train Set : [Download Mnist](https://wwr.lanzoui.com/iSBOeu43dkf)

## Training # 
To train a model, change the working directory to `./test`,then run: 

```bash
# 1p train perf
bash train_performance_1p.sh

# 8p train perf
bash train_performance_8p.sh

# 8p train full
bash train_full_8p.sh

# 8p eval
bash train_eval_8p.sh

# finetuning
bash train_finetune_1p.sh
```
After running,you can see the results in `./output`

## GAN training result # 

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 997      | 1        | 200      | O1       |
| -     | 11795     | 8        | 200      | O1       |



