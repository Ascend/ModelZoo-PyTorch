## Bert-base中文预训练模型训练方法

### 0.简介

基于huggingface transformers（https://github.com/huggingface/transformers）的bert base中文预训练模型，masked language 

modeling 任务。

### 1.安装依赖

```
pip3 install -r requirements.txt
```

### 2.安装transformers

```
cd transformers
pip3 install -e ./
cd ..
```

### 3.下载精度评估处理脚本

下载命令

```
curl https://raw.githubusercontent.com/huggingface/datasets/master/metrics/accuracy/accuracy.py -k -o accuracy.py
```

默认会下载accuracy.py到当前目录。如果将其下载到其他目录，请配置参数**--eval_metric_path**为accuracy.py的实际路径。

### 4.训练

#### Bert-base

下载配置模型和分词文件

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/bert-base-chinese
```

修改run_mlm_bertbase_1p.sh和run_mlm_bertbase_8p.sh中**--train_file**参数为使用的中文文本数据的实际路径，然后执行训练

单卡训练

```
bash run_mlm_bertbase_1p.sh
```

单机8卡训练

```
bash run_mlm_bertbase_8p.sh
```

#### Bert-large

下载配置模型和分词文件

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/algolet/bert-large-chinese
```

修改run_mlm_bertlarge_1p.sh和run_mlm_bertlarge_8p.sh中**--train_file**参数为使用的中文文本数据的实际路径，然后执行训练

单卡训练

```
bash run_mlm_bertlarge_1p.sh
```

单机8卡训练

```
bash run_mlm_bertlarge_8p.sh
```



### 附录：单机8卡训练脚本参数说明

```
python3 -m torch.distributed.launch --nproc_per_node 8 run_mlm.py \
        --model_type bert \                              # 模型类型
        --config_name bert-base-chinese/config.json \    # 模型配置文件
        --tokenizer_name bert-base-chinese \             # 分词文件路径
        --train_file ./train_huawei.txt \                # 数据集路径（会被自动分割为train和val两部分）
        --eval_metric_path ./accuracy.py \               # 精度评估处理脚本路径
        --line_by_line \                                 # 是否将数据中一行视为一句话
        --pad_to_max_length \                            # 是否对数据做padding处理
        --remove_unused_columns false \                  # 是否移除不可用的字段
        --save_steps 5000 \                              # 保存的step间隔
        --overwrite_output_dir \                         # 是否进行覆盖输出
        --per_device_train_batch_size 32 \               # 每个卡的train的batch_size
        --per_device_eval_batch_size 32 \                # 每个卡的evaluate的batch_size
        --do_train \                                     # 是否进行train
        --do_eval \                                      # 是否进行evaluate
        --fp16 \                                         # 是否使用混合精度
        --fp16_opt_level O2 \                            # 混合精度level
        --loss_scale 8192 \                              # loss scale值
        --use_combine_grad \                             # 是否开启tensor叠加优化
        --optim adamw_apex_fused_npu \                   # 优化器
        --output_dir ./output                            # 输出保存路径
```

### Q&A

1. Q:第一次运行报类似"xxx **socket timeout** xxx"的错误该怎么办？

   A:第一次运行tokenizer会对单词进行预处理，根据您的数据集大小，耗时不同，若时间过长，可能导致HCCL通信超时。此时可以通过设置以下环境变量，设置较大的超时时间阈值（单位秒，默认为600秒）：

   ```
   export HCCL_CONNECT_TIMEOUT=3600
   ```

   



