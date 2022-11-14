# ST-GCN 
This implements training of ST-GCN on the Kinetics-skeleton dataset, mainly modified from [mmskeleton](https://github.com/open-mmlab/mmskeleton/tree/master/deprecated/origin_stgcn_repo).

## ST-GCN Detail 

As of the current date, Ascend-Pytorch does not support einsum op and is still inefficient for contiguous operations . 
Therefore, ST-GCN is re-implemented using semantics such as custom OP. For details, see net/utils/tgcn.py . 

## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the Kinetics-skeleton dataset from author's [GoogleDrive](https://drive.google.com/open?id=103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb)
    - and extract files with
```
cd ST-GCN
unzip <path to st-gcn-processed-data.zip>
```

## Data Preparation

- Please follow this [link](https://github.com/open-mmlab/mmskeleton/tree/master/deprecated/origin_stgcn_repo#data-preparation) for preparing data. 

## Training 

To train a model, run `main.py` with the desired model architecture and the path to the Kinetics-skeleton dataset:

```bash
# 1p train 1p
bash test/train_full_1p.sh --data_path={data/path} # train accuracy

bash test/train_performance_1p.sh --data_path={data/path} # train performance

#  8p train 8p
bash test/train_full_8p.sh --data_path={data/path} # train accuracy

bash test/train_performance_8p.sh --data_path={data/path} # train performance

# 1p eval 1p
bash test/train_eval_1p.sh --data_path={data/path}

# onnx
python3.7.5 pthtar2onnx.py
```

## ST-GCN training result 

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| 31.62    | 46        | 1        | 50       | O2       |
| 31.62    | 293       | 8        | 50       | O2       |
