# GloRe 训练
# Graph-Based Global Reasoning Networks
This implements training of GloRe on the UCF-101 dataset.
- Reference implementation：
```
url=https: https://github.com/facebookresearch/GloRe
```

## GloRe Detail # 

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. 
Therefore, GloRe is re-implemented using semantics such as custom OP. 


## Requirements # 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
   Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- The UCF-101 Dataset can be downloaded from the links below.Move the datasets to directory ./dataset/UCF101/raw/data .
    - Train Set : [Download UCF-101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)
    - Test Set : [Download UCF-101](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)
- The pretrained model can be downloaded from the links below. Move the datasets to directory ./network/pretrain .
    - Pretrained model : [Download pth](https://dl.fbaipublicfiles.com/glore/kinetics/resnet50-lite_3d_8x8_w-glore_2-3_ep-0000.pth). Create directory ./network/pretrained/ and place pretrained model under directory ./network/pretrained/

## Training # 
To train a model, run `train_kinetics.py`:

```bash
# 1p train perf
bash test/train_performance_1p.sh

# 8p train perf
bash test/train_performance_8p.sh

# 8p train full
bash test/train_full_8p.sh

# finetuning
bash test/train_finetune_1p.sh
```

## GloRe training result # 

| ACC@1    | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| -        | 11.647      | 1        | 90      | O2       |
| 92.39     | 141.31     | 8        | 90      | O2       |



