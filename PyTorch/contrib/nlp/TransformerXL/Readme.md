# Transformer-xl

This implements training of transformer-xl on the enwik8 dataset, mainly modified from [pytorch/examples](https://github.com/kimiyoung/transformer-xl/tree/master/pytorch).

## Transformer-xl Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations.Therefore, Transformer-xl is re-implemented using semantics such as custom OP.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

## Data Prepration
- `bash getdata.sh`

## Training and Evaluation

To train a model, run `bash test/train_full_8p.sh` with the desired model architecture and the path to the enwik8 dataset:


```bash
#env
cd transformer-xl
dos2unix ./test/*.sh

# 1p train perf
bash test/train_performance_1p.sh

# 8p train perf
bash test/train_performance_8p.sh

# 8p train full
bash test/train_full_8p.sh

# 1p eval
bash test/eval_1p.sh

```

- 参数说明：
```bash
#--data               //数据集路径,可自行修改为对应路径的数据集
#--restart_dir        //加载模型checkpoint路径，可自行修改为对应路径的模型文件
#--addr               //主机地址 
#--max_step           //最大训练步数 
#--batch-size         //训练批次大小 
#--lr                 //初始学习率，默认：0.00025
#--device-list        //多卡训练指定训练用卡 ,8卡：'0,1,2,3,4,5,6,7'
#--amp                //是否使用混合精度 
#--loss-scale         //lossscale大小 
#--opt-level          //混合精度类型
```


## Transformer-xl training result

| bpc      | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 8300      | 1        | 1        | O2       |
| 1.09     | 44500     | 8        | 50       | O2       |



# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md