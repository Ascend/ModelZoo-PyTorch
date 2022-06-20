# Albert
A implementation of [A Lite Bert For Self-Supervised Learning Language Representations](https://arxiv.org/pdf/1909.11942.pdf)

base on [Albert-base-v2](https://github.com/lonePatient/albert_pytorch)

- add support for ascend npu, see run_classifier.py
- add fps counter, see tools/fps_counter.py
- add early stop, see run_classifier.py

## Result

- 精度性能

|   名称   | 精度  | 性能  |
| ---- | ---- | ---- | 
| NPU-1p   | 0.936  | 393  |
| NPU-8p   | 0.934 | 2675 |



## Before Run
- download raw [dataset](reference file README_raw.md in line 59 ) and [pretrained model](reference file README_raw.md in line 34)
- or simply get [albert_full](https://gitee.com/liuyj-suda-an/albert_full) 
- or only download [dataset](https://gitee.com/liuyj-suda-an/albert_full/tree/master/dataset) and [pretrained model](https://gitee.com/liuyj-suda-an/albert_full/tree/master/prev_trained_model) and [trained model](https://gitee.com/liuyj-suda-an/albert_full/tree/master/outputs) in [albert_full](https://gitee.com/liuyj-suda-an/albert_full)
- `pip install -r requirements.txt`
- `source ./test/env_npu.sh`
在模型当前目录创建数据集目录dataset,并放入数据集

## 自检
- 软件包

910版本
CANN toolkit：5.1.RC1
torch版本：1.8.1+ascend.rc2.20220505

- 自检脚本
```bash
# 1p train perf
# 是否正确输出了性能log文件
bash ./test/train_performance_1p.sh --data_path=./dataset

# 8p train perf
# 是否正确输出了性能log文件
bash ./test/train_performance_8p.sh --data_path=./dataset

# 8p train full
# 是否正确输出了性能精度log文件，是否正确保存了模型文件
bash ./test/train_full_8p.sh --data_path=./dataset

# 8p eval
# 是否正确输出了性能精度log文件
bash ./test/train_eval_8p.sh --data_path=./dataset

# online inference demo 
# 是否正确输出预测结果，请确保输入固定tensor多次运行的输出结果一致
python3.7 ./test/demo.py
```
