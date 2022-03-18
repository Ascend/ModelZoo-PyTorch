# FOTS: Fast Oriented Text Spotting with a Unified Network text detection branch reimplementation (PyTorch)
#  − 参考实现：
    ```
    url=https://github.com/Wovchena/text-detection-fots.pytorch 
    ```

## FOTS Detail

A unified end-to-end trainable Fast Oriented TextSpotting (FOTS) network for simultaneous detection andrecognition, sharing computation and visual information among the two complementary tasks.


## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- Install packages required
- Prepare dataset: 1. SynthText for pretrain
                   2. ICDAR2015 for finetune 

## Training

To train a model, run `train.py` with the desired model architecture
1. pretrain with SynthText for 9 epochs
2. finetune with ICDAR2015 for 583 epochs

```bash
# training 1p pretrain accuracy
bash ./test/train_full_pretrain_1p.sh 

# training 1p finetune accuracy
bash ./test/train_full_finetune_1p.sh

# training 1p performance
bash ./test/train_performance_1p.sh 

# training 8p pretrain accuracy
bash ./test/train_full_pretrain_8p.sh 

# training 8p finetune accuracy
bash ./test/train_full_finetune_8p.sh 

# training 8p performance
bash ./test/train_performance_8p.sh 

#test 8p accuracy
bash ./test/test_8p.sh 

#test 1p accuracy
bash ./test/test_1p.sh 
```

Log path:
    FOTS/*.log


## FOTS training result

| Acc@1    | Recall    | Hmean    | FPS      | Npu_nums | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: | :------: | :------: |
| -        | -         | -        | 16.101   | 1        | 20       |   O2     |
| 85.046   | 78.864    | 81.838   | 77.614   | 8        | 583      |   O2     |