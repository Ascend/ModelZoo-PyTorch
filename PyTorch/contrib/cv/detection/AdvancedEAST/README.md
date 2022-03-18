# AdvancedEAST

实现了AdvancedEAST在天池ICPR数据集上的训练。
- 参考实现：
```
url=https://github.com/BaoWentz/AdvancedEAST-PyTorch
branch=master 
commit_id=a835c8cedce4ada1bc9580754245183d9f4aaa17
```

## AdvancedEAST Detail

- 为数据集前处理增加了多线程优化
- 增加了混合精度训练
- 增加了多卡分布式训练
- 增加了CosineAnnealingLR
- 优化了loss在NPU上的计算效率

## Requirements

- CANN 5.0.2及对应版本的PyTorch
- `pip install -r requirements.txt`
- 下载[天池ICPR数据集](https://pan.baidu.com/s/1NSyc-cHKV3IwDo6qojIrKA)，密码: ye9y
    - 下载ICPR_text_train_part2_20180313.zip和[update] ICPR_text_train_part1_20180316.zip两个压缩包，新建目录icpr和子目录icpr/image_10000、icpr/txt_10000，将压缩包中image_9000、image_1000中的图片文件解压至image_10000中，将压缩包中txt_9000、txt_1000中的标签文件解压至txt_10000中
    - `bash test/prep_dataset.sh`

## Training

依次训练size为256x256，384x384，512x512，640x640，736x736的图片，每个size加载上个size的训练结果，加速模型收敛。

```bash
# 1p train perf
bash test/train_performance_1p.sh

# 8p train perf
bash test/train_performance_8p.sh

# 8p train full
bash test/train_full_8p.sh
# 默认依次训练256，384，512，640，736五个size，可以指定要训练size，用于恢复中断的训练，例如
# bash test/train_full_8p.sh 640 736

# eval
bash test/train_eval.sh
# 默认评估736 size，可以指定要评估的size，例如
# bash test/train_eval.sh 640

# finetuning
bash test/train_finetune_1p.sh

# online inference demo 
python3.7 demo.py

# To ONNX
python3.7 pth2onnx.py
```

## AdvancedEAST training result

| Size     | F1-score | FPS       | Npu_nums | Epochs   | AMP_Type |
| :------: | :------: | :------:  | :------: | :------: | :------: |
| 256      | -        | 254       | 1        | -        | O1       |
| 256      | -        | 1075      | 8        | 60       | O1       |
| 384      | -        | 118       | 1        | -        | O1       |
| 384      | -        | 680       | 8        | 60       | O1       |
| 512      | -        | 63        | 1        | -        | O1       |
| 512      | -        | 400       | 8        | 60       | O1       |
| 640      | -        | 37        | 1        | -        | O1       |
| 640      | -        | 243       | 8        | 60       | O1       |
| 736      | -        | 34        | 1        | -        | O1       |
| 736      | 62.41%   | 218       | 8        | 60       | O1       |
