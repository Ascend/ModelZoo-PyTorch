# Machine Translation with Transformer

## Requirements
* NPU配套的run包安装
* Python 3.7.5
* PyTorch(NPU版本)
* apex(NPU版本)
* dllogger


## Dataset Prepare
1. 运行sh run_preprocessing.sh下载数据集，并处理

## 1P
1. 编辑 train_1p.sh device-id(NPU设备号)  DATA_DIR(数据集目录) MODELDIR(日志和模型保存目录)
2. 运行 sh train_1p.sh
```
python3 -u train_1p.py \
  ./data/dataset/wmt14_en_de_joined_dict/ \
  --device-id 7\
  --arch transformer_wmt_en_de \
  --share-all-embeddings \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.997 \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000 \
  --lr 0.0006 \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-sentences 128\
  --max-tokens 102400\
  --seed 1 \
  --save-dir $MODELDIR \
  --save-interval 1\
  --update-freq 8\
  --log-interval 1\
  --stat-file $STAT_FILE\
  --distributed-world-size 1\
  --amp\
  --amp-level O2

```
## 8P
1. 编辑 train_8p.sh device-id(NPU设备号)  DATA_DIR(数据集目录) MODELDIR(日志和模型保存目录) addr(本机设备ip)
2. 运行 sh train_8p.sh

```

python3 train_np.py $DATA_DIR \
  --arch transformer_wmt_en_de \
  --share-all-embeddings \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.997 \
  --addr 'XX.XXX.XXX.XXX' \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000 \
  --lr 0.0006 \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-sentences 128\
  --max-tokens 102400 \
  --seed 1 \
  --save-dir $MODELDIR \
  --stat-file $STAT_FILE\
  --log-interval 1\
  --amp\
  --amp-level O2

```

