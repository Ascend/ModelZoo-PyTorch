# SpanBERT

This implements training of SpanBERT on the SQuAD1.1 dataset, mainly modified from [SpanBERT](https://github.com/facebookresearch/SpanBERT).

## SpanBERT Detail

SpanBERT  improves the performance of BERT by introducing a span-level pretraining approach(which differs from BERT in both the masking scheme and the training objectives.)

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the SQuAD1.1 dataset from https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset
- Confirm your dataset
```
      ---data
         ---train-v1.1.json
         ---dev-v1.1.json
```

## Training

To train a model, run `./code/run_squad.py` with the desired model architecture and the path to the SQuAD1.1 dataset:

```bash

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=data

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=data

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=data

```

Log path:
    test/output/devie_id/train_${device_id}.log           # training detail log
    test/output/devie_id/SpanBert_bs32_8p_perf.log  # 8p training performance result log
    test/output/devie_id/train_SpanBert_bs32_8p_perf_loss.txt   # 8p training loss result log



## SpanBERT training result

| F1       | FPS       | Npu_nums | Epochs   | AMP_Type | loss scale |
| :------: | :------:  | :------: | :------: | :------: | :------:   |
| -        | 24.3      | 1        | 4        | O2       |   128.0    |
| 91.9     | 47.2      | 8        | 4        | O2       |   128.0    |

## Additional 
Please download 'dict.txt' from https://github.com/facebookresearch/SpanBERT/tree/main/pretraining
and put it in 'SpanBERT/pretraining'
