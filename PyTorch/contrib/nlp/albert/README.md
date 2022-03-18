# Albert
A implementation of [A Lite Bert For Self-Supervised Learning Language Representations](https://arxiv.org/pdf/1909.11942.pdf)

base on [Albert-base-v2](https://github.com/lonePatient/albert_pytorch)

- add support for ascend npu, see run_classifier.py
- add fps counter, see tools/fps_counter.py
- add early stop, see run_classifier.py

## Result
- 软件包

803版本
CANN toolkit：5.0.2
torch版本：1.5.0+ascend.post3
固件驱动21.0.2

- 精度性能

|   名称   | 精度  | 性能  |
| ---- | ---- | ---- | 
| GPU-1p   | 0.927  | 517  |
| GPU-8p   | 0.936 | 3327 |
| NPU-1p   | 0.927  | 400  |
| NPU-8p   | 0.927 | 2605 |



## Before Run
- download raw [dataset](reference file README_raw.md in line 59 ) and [pretrained model](reference file README_raw.md in line 34)
- or simply get [albert_full](https://gitee.com/liuyj-suda-an/albert_full) 
- or only download [dataset](https://gitee.com/liuyj-suda-an/albert_full/tree/master/dataset) and [pretrained model](https://gitee.com/liuyj-suda-an/albert_full/tree/master/prev_trained_model) and [trained model](https://gitee.com/liuyj-suda-an/albert_full/tree/master/outputs) in [albert_full](https://gitee.com/liuyj-suda-an/albert_full)
- `pip install -r requirements.txt`
- `source ./test/env_npu.sh`
在模型当前目录创建数据集目录dataset,并放入数据集

## 自检
- 软件包

803版本
CANN toolkit：5.0.2
torch版本：1.5.0+ascend.post3
固件驱动21.0.2

- 自检脚本
```bash
# 1p train perf
# 是否正确输出了性能log文件
bash ./test/train_performance_1p.sh --data_path=./dataset
# 验收结果：OK
# 备注：验收测试性能400fps，超过GPU-1p性能 517fps的1/2，输出日志在./test/output/SST-2.log

# 8p train perf
# 是否正确输出了性能log文件
bash ./test/train_performance_8p.sh --data_path=./dataset
# 验收结果：OK
# 备注：验收测试性能2605fps，超过GPU-8p性能 3327fps的1/2，输出日志在./test/output/SST-2.log

# 8p train full
# 是否正确输出了性能精度log文件，是否正确保存了模型文件
bash ./test/train_full_8p.sh --data_path=./dataset
# 验收结果：OK
# 备注：目标精度0.926，验收测试精度0.927，输出日志在./test/output/SST-2.log，模型保存在./outputs/SST-2下

# 8p eval
# 是否正确输出了性能精度log文件
bash ./test/train_eval_8p.sh --data_path=./dataset
# 验收结果：OK
# 备注：输出日志在./test/output/SST-2.log

# finetuning
# 是否正确执行迁移学习
bash ./test/train_finetune_1p.sh --data_path=./dataset
# 验收结果：OK
# 备注：功能正确，输出日志在./test/output/STS-B.log

# online inference demo 
# 是否正确输出预测结果，请确保输入固定tensor多次运行的输出结果一致
python3.7 ./test/demo.py
# 验收结果：OK
# 备注：功能正确，无输出日志
```
