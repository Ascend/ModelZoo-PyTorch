

# deepspeech.pytorch

This implements training of deepspeech on NPU mainly modified from [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)

## installation

### From Source

install this fork Warp-CTC bindings:

```shell
### npu环境变量
source {deepspeech_root}/scripts/env_new.sh
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
git checkout -b pytorch_bindings origin/pytorch_bindings
mkdir build; cd build; cmake ..; make
cd ../pytorch_binding && python3.7 setup.py install
```

install requirements

```shell
pip3 install -r requirements.txt
```

If you plan to use Multi-GPU/Multi-node training, you'll need etcd. Below is the command to install on Ubuntu.

```shell
sudo apt-get install etcd
sudo apt-get install sox
```

## Training

### Download Dataset

All you need is entering the data directory and execute the follow scripts

```shell
cd data
python3.7 an4.py
```

### Training a Model

#### 1p training

```shell
# The result will be placed in the current directory 1p_train.log
bash run_1p.sh
```

#### 8p training

```shell
# The result will be placed in the current directory 8p_train.log
bash run_8p.sh
```

### Performance

```shell
### 1p performance, the log will be placed in the current directory 1p_train_performance.log
bash train_performance_1p.sh
### 8p performance, the log will be placed in the current directory 8p_train_performance.log
bash train_performance_8p.sh
```

## Testing/Inference

To evaluate a trained model on a test set (has to be in the same format as the training set):

```shell
# if you want to see the final precision, you can execute the follow scripts after execute 1p or 8p training scripts
bash eval.sh
```

## Result

|       |  WER   |  CER   | Epochs | APEX | FPS  |
| :---: | :----: | :----: | :----: | :--: | :--: |
| NPU1P | 9.444  | 5.723  |   70   |  O2  |  4   |
| NPU8P | 17.464 | 10.926 |   70   |  O2  |  22  |
| GPU1P | 10.349 | 7.076  |   70   |  O2  |  94  |
| GPU8P | 15.265 | 9.834  |   70   |  O2  | 377  |

