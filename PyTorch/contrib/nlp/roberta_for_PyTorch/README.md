## RoBERTa

This implements training of RoBERTa on the SST-2 dataset, mainly modified from [pytorch/fairseq](https://github.com/pytorch/fairseq).

## Requirements

- NPU配套的run包安装
- Python 3.7.5
- PyTorch(NPU版本)
- apex(NPU版本)

## Enviroment preparation

Please refer to the [pytorch/fairseq](https://github.com/pytorch/fairseq)

## dataset 
Please refer to examples/roberta/preprocess_GLUE_tasks.sh

## Train model

```bash
source ./test/env_npu.sh

# single card training on npu
bash ./test/train_1p_npu.sh

# mutil card training on npu
bash ./test/train_8p_npu.sh

# single card training on gpu
bash ./test/train_1p_gpu.sh

# mutil card training on gpu
bash ./test/train_8p_gpu.sh

# Won't save log and trained model
```

## Result

|  名称  | Epoch | 性能 | 精度  |
| :----: | :---: | :--: | :---: |
| GPU-1p |   1   | 397  | 0.927 |
| GPU-8p |  10   | 2997 | 0.943 |
| NPU-1p |   1   | 216  | 0.938 |
| NPU-8p |  10   | 1985 | 0.969 |

## Check
```bash
# 1p train perf
bash test/train_performance_1p.sh
# 备注:输出日志在./output/1p_npu_performance.log

# 8p train perf
bash test/train_performance_8p.sh
# 备注:输出日志在./output/8p_npu_performance.log

# 8p train full
bash test/train_full_8p.sh
# 备注:输出日志在./output/8p_npu_full.log，模型保存在./output/checkpoints下

# 8p eval
# 是否正确输出了性能精度log文件
bash test/train_eval_8p.sh
# 备注：输出日志在./output/eval.log

# finetuning
bash test/train_finetune_1p.sh
# 备注:输出日志在./output/1p_npu_finetune.log

# online inference demo 
# 是否正确输出预测结果，请确保输入固定tensor多次运行的输出结果一致
python3.7.5 test/demo.py
