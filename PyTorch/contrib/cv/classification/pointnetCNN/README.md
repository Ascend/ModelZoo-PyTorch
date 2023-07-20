# PointNetCNN

This implements training of PointNetCNN on the modelnet40_ply_hdf5_2048 dataset.

## PointNetCNN Detail



## Requirements

- `pip install -r requirements.txt`
- Download  the  dataset from  https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E6%8E%A8%E7%90%86/cv/classfication/PointNetCNN/modelnet40_ply_hdf5_2048.zip  and unzip it to ./data
  
## Training

To train a model, run `train_pytorch.py` 

```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=./data/modelnet40_ply_hdf5_2048/

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=./data/modelnet40_ply_hdf5_2048/

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=./data/modelnet40_ply_hdf5_2048/

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=./data/modelnet40_ply_hdf5_2048/


# finetuning 1p 
bash test/train_finetune_1p.sh --data_path=./data/modelnet40_ply_hdf5_2048/
```

Log path:
    ./test/output/1/train_1.log           # 8p training performance and loss log
    ./test/output/1/train_1.log           # 1p training performance and loss log




## PointNetCNN training result

| Acc@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 20       | 1        | 1        | O1       |
| -        | 160      | 8        | 250      | O1       |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md