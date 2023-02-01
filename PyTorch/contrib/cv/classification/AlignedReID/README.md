# AlignedReID

This implements training of AlignedReID on the ImageNet dataset.
- Reference implementation：
```
url=https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch
branch=master 
commit_id=2e2d45450d69a3a81e15d18fe85c2eebbde742e4 
```

## AlignedReID Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore, AlignedReID is re-implemented using semantics such as custom OP. 


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the [market1501 dataset](https://drive.google.com/drive/folders/1CaWH7_csm9aDyTVgjs7_3dlZIWqoBlv4)
    - Then, unzip the images.tar file
    
## Training

To train a model, run `main_1p.py` and `main_8p.py`  with the desired model architecture and the path to the market1501 dataset:

```bash
# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path

# training 8p accuracy, pth file will be saved in the current path
bash ./test/train_full_8p.sh --data_path=real_data_path

#test 8p accuracy
bash test/train_eval_8p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path 

# finetuning 1p, input other cunstomed pkl file by adding pkl_path parameter to test finetuning function
bash test/train_finetune_1p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path --pkl_path=customized_index_file_path

# online inference demo 
python3.7 demo.py
```

## AlignedReID training result

| rank@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| 83.02        | 80      | 1        | 300      | O1       |
| 82.16     | 620     | 8        | 300      | O1       |


