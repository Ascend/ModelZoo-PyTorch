## Bert-base中文预训练模型训练方法

## 0.安装依赖

```
pip3 install -r requirements.txt
```

### 1.下载模型配置和分词配置文件

在当前目录执行下载命令

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/bert-base-chinese
```

下载后会在当前目录生成bert-base-chinese子目录

### 2.训练

修改run_mlm_cn.sh和run_mlm_cn_8p.sh中**--train_file**参数为使用的中文文本数据的实际路径，然后执行训练

单卡训练

```
bash run_mlm_cn.sh
```

单机8卡训练

```
bash run_mlm_cn_8p.sh
```

