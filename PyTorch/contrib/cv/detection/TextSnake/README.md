# Textsnake

This implements training of Textsnake on the TotalText dataset, whose reference implementation is https://github.com/princewang1994/TextSnake.pytorch.

## TextSnake Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. Therefore, TextSnake is re-implemented using semantics such as custom OP.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- Install geos (Choose one from three methods)
    - apt-get install libgeos-dev
    - yum install geos-devel
    - Source Code Install
        - wget http://download.osgeo.org/geos/geos-3.8.1.tar.bz2
        - bunzip2 geos-3.8.1.tar.bz2
        - tar xvf geos-3.8.1.tar
        - cd geos-3.8.1
        - ./configure && make && make install

- `pip install -r requirements.txt`
- Download the TotalText dataset
    - HomePage: https://github.com/cs-chan/Total-Text-Dataset
    - download dataset and make it to a dataset format
    - put the dataset into ./data

```shell
cd ./dataset/total_text
bash download.sh
```
- Download the pretrained model and put it to ./save/synthtext_pretrain
    - https://pan.baidu.com/s/1Q4D3pDyVP7qdi2Cs-vc9cQ (extract code: xmoh)
    
## Training

To train a model, run `train_textsnake_npu.py` with the desired model architecture and the path to the TotalText dataset:

```bash


# 1p train perf
bash test/train_performance_1p.sh 

# 8p train perf
bash test/train_performance_8p.sh

# training 1p accuracy
bash ./test/train_full_1p.sh 

# 8p train full
bash test/train_full_8p.sh

# finetuning
bash test/train_finetune_1p.sh

#test 8p accuracy
bash test/train_eval_8p.sh 


```



## Textsnake training result

| Precision |   Recall  | F-measure | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------:  | :------:  | :------:  | :------:  | :------: | :------: | :------: |
| -         | -         | -         | -         | 1        | 1        | O1       |
| 0.741     | 0.687     | 0.713     | 29        | 8        | 200      | O1       |
