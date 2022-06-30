# MAE

This implements training of MAE on the ImageNet dataset, mainly modified from [MAE](https://github.com/facebookresearch/mae).

## MAE Detail

1. 迁移到 NPU 上
2. 支持分布式训练和数据并行
3. 使用apex进行混合精度训练

## Requirements

- `pip install -r requirements.txt`
- 说明：requirements.txt里面的torch和apex版本均为linux_aarch64架构包
- 下载ImageNet数据集

## Pre-Training

- To pretrain a MAE model based on ViT-Base, run `main_pretrain.py` 

- with the path to the ImageNet dataset.（`real_data_path` is a directory containing `{train, val}` sets of ImageNet）

```bash
# pre-training 1p performance，单p上运行1个epoch，运行时间约为1h
# 输出性能日志./output_pretrain_1p/910A_1p_pretrain.log、总结性日志./output_pretrain_1p/log.txt
bash ./test/pretrain_performance_1p.sh --data_path=real_data_path

# pre-training 8p performance，8p上运行1个epoch，运行时间约为9min
# 输出性能日志./output_pretrain_8p/910A_8p_pretrain.log、总结性日志./output_pretrain_8p/log.txt
bash ./test/pretrain_performance_8p.sh --data_path=real_data_path

# pre-training 8p full，8p上运行400个epoch，运行时间约为60h
# 输出完整预训练日志./output_pretrain_full_8p/910A_8p_pretrain_full.log、总结性日志./output_pretrain_full_8p/log.txt
bash ./test/pretrain_full_8p.sh --data_path=real_data_path
```

- 注：在1p脚本`./test/pretrain_performance_1p.sh`中，可以修改变量local_rank，指定NPU预训练，默认`local_rank=0`

## MAE Pre-Training Result

| NAME | LOSS | FPS | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| 1p-GPU  | -      | 320   | 1        | -       |
| 1p-NPU | -     | 328  | 1      | O2      |
| 8p-GPU | 0.4107 | 2399 | 400 | - |
| 8p-NPU | 0.4107 | 2515 | 400 | O2 |

## Fine-Tuning

- To fine-tune a ViT-Base model from the MAE pretrained model, run `main_finetune.py` 
- with the path to the ImageNet dataset.（`real_data_path` is a directory containing `{train, val}` sets of ImageNet）
- with the path to the pretrained model.（`pretrained_model_path` is the path to the MAE model you want to fine-tune）
- For 8p eval，`finetuned_model_path`  is the path to the finetuned model you want to evaluate.

```bash
# fine-tuning 1p performance，单p上运行1个epoch，运行时间约为1h15min，
# 输出性能日志./output_finetune_1p/910A_1p_finetune.log、总结性日志./output_finetune_1p/log.txt
bash ./test/finetune_performance_1p.sh --data_path=real_data_path --finetune_pth=pretrained_model_path

# fine-tuning 8p performance，8p上运行1个epoch，运行时间约为11min
# 输出性能日志./output_finetune_8p/910A_8p_finetune.log、总结性日志./output_finetune_8p/log.txt
bash ./test/finetune_performance_8p.sh --data_path=real_data_path --finetune_pth=pretrained_model_path

# fine-tuning 8p full，8p上运行100个epoch，运行时间约为18h
# 输出完整微调日志./output_finetune_full_8p/910A_8p_finetune_full.log、总结性日志./output_finetune_full_8p/log.txt
bash ./test/finetune_full_8p.sh --data_path=real_data_path --finetune_pth=pretrained_model_path

# 8p eval，运行时间约为3min
# 输出eval日志./output_finetune_eval_8p/910A_8p_finetune_eval.log
bash ./test/finetune_eval_8p.sh --data_path=real_data_path --resume_pth=finetuned_model_path
```

- 注：在1p脚本`./test/finetune_performance_1p.sh` 中，可以修改变量`local_rank`，指定NPU微调，默认`local_rank=0`

## MAE Fine-Tuning Result
| NAME | Acc@1 | FPS | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: | :------: |
| 1p-GPU  | -      | 218   | 1        | -       |
| 1p-NPU | -     | 306   | 1      | O2      |
| 8p-GPU | 83.07 | 1538 | 100 | - |
| 8p-NPU | 83.34 | 2263 | 100 | O2 |