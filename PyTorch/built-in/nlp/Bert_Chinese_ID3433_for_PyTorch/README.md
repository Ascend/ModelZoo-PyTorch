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

### 3.训练

#### （可选）数据集预处理

如果你想重新处理zhwiki的原始数据，可按照以下步骤操作。

下载zhwiki

```
wget https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2 --no-check-certificate
```

解压得到zhwiki-latest-pages-articles.xml

```
bzip2 -dk zhwiki-latest-pages-articles.xml.bz2
```

使用工程目录的WikiExtractor.py提取文本，其中extracted/wiki_zh为保存路径，不要修改

```
python3 WikiExtractor.py zhwiki-latest-pages-articles.xml -b 100M -o extracted/wiki_zh
```

将多个文档整合为一个txt文件，在本工程根目录下执行

```
python3 WikicorpusTextFormatting.py --extracted_files_path extracted/wiki_zh --output_file zhwiki-latest-pages-articles.txt
```

最终生成的文件名为zhwiki-latest-pages-articles.txt

#### Bert-base

下载配置模型和分词文件

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/bert-base-chinese
```

将下载下的bert-base-chinese放置在模型脚本一级目录下

单卡训练

```
bash test/train_full_1p.sh --data_path=dataset_file_path --batch_size=32 --model_size=base  --device_id=0  # 单卡精度训练
bash test/train_performance_1p.sh --data_path=dataset_file_path --batch_size=32 --model_size=base    # 单卡性能训练
```

单机8卡训练

```
bash test/train_full_8p.sh --data_path=dataset_file_path --batch_size=32 --model_size=base    # 8卡精度训练
bash test/train_performance_8p.sh --data_path=dataset_file_path --batch_size=32 --model_size=base    # 8卡性能训练
```
```
训练脚本参数说明：
	--data_path：  数据集路径
	--model_size： 训练model是base或者是large
	--device_id：  单卡训练时所使用的device_id

双机16卡训练
```

```
bash test/train_full_16p.sh --data_path=dataset_file_path --batch_size=32 --model_size=base --node_rank=node_id --master_addr=x.x.x.x --master_port=xxxx # 8卡精度训练    # 8卡精度训练
bash test/train_performance_16p.sh --data_path=dataset_file_path --batch_size=32 --model_size=base --node_rank=node_id --master_addr=x.x.x.x --master_port=xxxx # 8卡精度训练   # 8卡性能训练
```

```
训练脚本参数说明：
	--data_path：  数据集路径
	--model_size： 训练model是base或者是large
	--device_id：  单卡训练时所使用的device_id
	--node_rank:   集群节点序号，master节点是0， 其余节点依次加1
	--master_addr：master节点服务器的ip
	--master_port: 分布式训练中,master节点使用的端口
```

#### Bert-large

下载配置模型和分词文件

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/algolet/bert-large-chinese
```

将下载下的bert-large-chinese放置在模型脚本一级目录下

单卡训练

```
bash test/train_full_1p.sh --data_path=dataset_file_path --batch_size=16 --model_size=large  --device_id=0   # 单卡精度训练
bash test/train_performance_1p.sh --data_path=dataset_file_path --batch_size=16 --model_size=large    # 单卡性能训练
```

单机8卡训练

```
bash test/train_full_8p.sh --data_path=dataset_file_path --batch_size=16 --model_size=large    # 8卡精度训练
bash test/train_performance_8p.sh --data_path=dataset_file_path --batch_size=16 --model_size=large    # 8卡性能训练
```

```
训练脚本参数说明：
	--data_path：  数据集路径
	--model_size： 训练model是base或者是large
	--device_id：  单卡训练时所使用的device_id
```

双机16卡训练

```
bash test/train_full_16p.sh --data_path=dataset_file_path --batch_size=32 --model_size=large --node_rank=node_id --master_addr=x.x.x.x --master_port=xxxx # 8卡精度训练    # 8卡精度训练
bash test/train_performance_16p.sh --data_path=dataset_file_path --batch_size=32 --model_size=large --node_rank=node_id --master_addr=x.x.x.x --master_port=xxxx # 8卡精度训练   # 8卡性能训练
```

```
训练脚本参数说明：
	--data_path：  数据集路径
	--model_size： 训练model是base或者是large
	--device_id：  单卡训练时所使用的device_id
	--node_rank:   集群节点序号，master节点是0， 其余节点依次加1
	--master_addr：master节点服务器的ip
	--master_port: 分布式训练中,master节点使用的端口
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

   A:第一次运行tokenizer会对单词进行预处理，根据您的数据集大小，耗时不同，若时间过长，可能导致等待超时。此时可以通过设置较大的超时时间阈值尝试解决：

   （1）设置pytorch框架内置超时时间，修改脚本中的distributed_process_group_timeout（单位秒）为更大的值，例如设置为7200：
   
```
    --distributed_process_group_timeout 7200
```

   （2）设置HCCL的建链时间为更大的值，修改env.sh中环境变量HCCL_CONNECT_TIMEOUT（单位秒）的值：

```
   export HCCL_CONNECT_TIMEOUT=7200
```
2. Q:如果训练报wandb.error.UsageError:api_key not configured (no-tty)的错误该怎么办?
  
   A:export WANDB_DISABLED=1



