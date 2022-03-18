# RawNet2

This implements training of RawNet2 on the VoxCeleb1&2 datasets of YouTube.

- Reference implementation:

```
url=https://github.com/Jungjee/RawNet
dir=RawNet-master/python/RawNet2 
```

## Baseline-RawNet2 Detail

As of the current date, Ascend-Pytorch is still inefficient for contiguous operations. Therefore, RawNet2  is re-implemented using semantics such as custom OP. 

## Requirements 

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

## DataSet

```
url: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
```

- The training datasets are VoxCeleb2, the evaluation dataset is VoxCeleb1H & VoxCeleb1E. The datasets are large. Please ensure sufficient hard disk space when downloading and decompressing.
- Besides, the data in the VoxCeleb2 downloaded from the url above is in the format of .m4a. If you do not use the dataset which is converted already, you should firstly run the `m4a2wav.py`.
- You need to follow directory structure of the `data` as below. If you connect to the prepared data folder, you don't need to build the following directory.

```
${RawNet}/DB/VoxCeleb1/
├── dev_wav
│   ├── id10001
│   │   ├── 1zcIwhmdeo4
│   │   │   ├── 00001.wav
│   │   │   ├── 00002.wav
│   │   │   └── 00003.wav
│   │   ├── 5ssVY9a5X-M
│   │   │   ├── 00001.wav
│   │   │   ├── 00002.wav
│   │   │   ├── 00003.wav
│   │   │   └── 00003.wav
│   └── ...
├── eval_wav
│   ├── id10270
│   │   ├── 5r0dWxy17C8
│   │   │   ├── 00001.wav
│   │   │   ├── 00002.wav
│   │   │   ├── 00003.wav
│   │   │   ├── 00004.wav
│   │   │   └── 00005.wav
│   └── ...
│       ├── _z_BR0ERa9g
│           ├── 00001.wav
│           ├── 00002.wav
│           └── 00003.wav
├── val_trial.txt
└── veri_test.txt 

${RawNet}/DB/VoxCeleb2/
└── wav
    ├── id00012
    │   ├── 21Uxsk56VDQ
    │   │   ├── 00001.wav
    │   │   ├── ...
    │   │   └── 00059.wav
    │   ├── 00-qODbtozw
    │   │   ├── ...
    │   │   ├── 00079.wav
    │   │   └── 00080.wav
    ├── ...
    │   └── zw-4DTjqIA0
    │       ├── 00108.wav
    │       └── 00109.wav
    └── id09272
        └── u7VNkYraCw0
            ├── ...
            └── 00027.wav
```

- You need to follow directory structure of the `output` as below.

```
${RawNet}/train/train_${device_count}P
|-- DNNS/${name}/
|   |-- models
|   |   |--best_opt_eval.pt ## The best perfomance model is saved here
|   |   |--TA_${epoch}_${eer}.pt ##The other model is saved here
|   |-- results
|   |-- log
|   |   |-- eval_epoch${epoch}.txt   ## The training log is saved here
|   |-- prof
|   |-- eers.txt  ##The eers is saved here
|   |-- f_params.txt ##The params of the model are saved here
```

## Training #

- Note that the `output` folder under the `test` directory will also save the code running log.
- To run the model, you should cd to the directory of test
- To train a model, run `train_1p.py` or `train_8p.py`: 

```bash
# 1p train perf
nohup bash train_performance_1p.sh --data_path=xxx &

# 8p train perf
nohup bash train_performance_8p.sh --data_path=xxx &

# 1p train full
nohup bash train_full_1p.sh --data_path=xxx &

# 8p train full
nohup bash train_full_8p.sh --data_path=xxx &

```

## RawNet2 training result 

|       eer(percentage)       | FPS(aver) | Npu_nums | Epochs | AMP_Type |
| :-------------------------: | :-------: | :------: | :----: | :------: |
|            0.14             |   7760    |    1     |   1    |    O2    |
| 0.038(aver) and 0.035(high) |   8573    |    8     |   20   |    O2    |

### **Testing**

The testing data in the paper is about the VoxCeleb1H and VoxCeleb1E. And here we use the dataset of the VoxCeleb1H, and the target of the eer in the paper is 0.0489.
