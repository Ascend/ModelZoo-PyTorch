TinyBERT
======== 
Welcome to the TinyBERT project! Please read the following instructions carefully so as to reproduce the project better.

Introduction
===========
This implement training of TinyBERT on the SST-2 dataset is mainly modified from the following [link](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT).

For more details about the techniques of TinyBERT, refer to the paper: [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)

Specifically, this implement is modified to adapt the NPU chips.

Requirements
===========
- working dir

First of all, you need to use the command ```cd``` to change the current working dir to where the test folder locates.

- virtual environment

Run command below to install the environment(**using python3**)
```bash
pip3.7 install -r requirements.txt
# or
conda install --yes --file requirements.txt
```
- dataset

TinyBERT is trained on the dataset SST-2, and we also apply TinyBERT to transfer learning on MNLI dataset. You can get the dataset by running the command:
```
wget https://ascend-pytorch-one-datasets.obs.cn-north-4.myhuaweicloud.com/train/zip/SST-2-TinyBert.zip
```

- model

Three models are required in the project.

The first one is the teacher model, which is the BERT-base-uncased model finetuned on SST-2. The second one is the other teacher model, which is the BERT-base-uncased model finetuned on MNLI(only for transfer learning). And the third one is the student model. We adopt the general-distilled model(4layer-312dim) provided by Huawei Noah's Ark Lab.

You can download the pretrained-model files by running the commands:
```
# download the student model
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E8%AE%AD%E7%BB%83/nlp/TinyBERT/%E6%A8%A1%E5%9E%8B%E6%96%87%E4%BB%B6/%E3%80%90%E8%AE%AD%E7%BB%83%E3%80%91%E5%AD%A6%E7%94%9F%E6%A8%A1%E5%9E%8B.zip

# download the teacher model(finetuned on MNLI dataset)
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E8%AE%AD%E7%BB%83/nlp/TinyBERT/%E6%A8%A1%E5%9E%8B%E6%96%87%E4%BB%B6/%E3%80%90%E8%AE%AD%E7%BB%83%E3%80%91%EF%BC%88%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%EF%BC%89MNLI%E6%95%99%E5%B8%88%E6%A8%A1%E5%9E%8B.zip

# download the teacher model(finetuned on SST-2 dataset)
wget https://ascend-pytorch-model-file.obs.cn-north-4.myhuaweicloud.com/%E9%AA%8C%E6%94%B6-%E8%AE%AD%E7%BB%83/nlp/TinyBERT/%E6%A8%A1%E5%9E%8B%E6%96%87%E4%BB%B6/%E3%80%90%E8%AE%AD%E7%BB%83%E3%80%91%EF%BC%88%E6%AD%A3%E5%BC%8F%E8%AE%AD%E7%BB%83%EF%BC%89SST-2%E6%95%99%E5%B8%88%E6%A8%A1%E5%9E%8B.zip
```

Training
===================
To train a model, run main.py with the desired model architecture. Unlike other one-step model, there are two training processes in task-distillation of TinyBERT model.

*Please pay attention: all of the performance scripts are set to stop running when having run 1000 steps, for they are just designed to test whether the code works and the files can be exported. There will be a tip like:"End performance testing. Ready to exit". It's a normal phenomenon instead of a bug. Please ignore it and just go on following the instructions and reproducing the project.*

Establish empty directory
-------------------------
```
# make directory
mkdir tmp_tinybert_performance
mkdir tmp_tinybert_dir
mkdir TinyBERT_dir
mkdir TinyBERT_dir_performance
mkdir output
# set the authority(use sudo if necessary)
chmod 777 tmp_tinybert_performance
chmod 777 tmp_tinybert_dir
chmod 777 TinyBERT_dir
chmod 777 TinyBERT_dir_performance
chmod 777 output
```

1p mode
-------

```
# Step 1: run the intermediate layer distillation.
bash ./test/train_performance_1p_1.sh
bash ./test/train_full_1p_1.sh

# Step 2: run the prediction layer distillation. 
bash ./test/train_full_1p_2.sh

# Step 3: run the evaluation on the SST-2 dataset
bash ./test/train_eval_1p.sh      
```

8p mode
-------
```
# Step 1: run the intermediate layer distillation.
bash ./test/train_performance_8p_1.sh
bash ./test/train_full_8p_1.sh

# Step 2: run the prediction layer distillation. 
bash ./test/train_full_8p_2.sh

# Step 3: run the evaluation on the SST-2 dataset
bash ./test/train_eval_8p.sh    
```

Other setting
-------------
```
# Transfer learning
bash ./test/train_finetune_1p.sh
# demo(automatically repeat 20 times)
bash ./test/demo.sh
```

After finishing the whole training process, you can see all output files in the directory ./output

Result
======
|device|acc of 1p|acc of 8p|fps of 1p|fps of 8p 
|  ----  | ----  | ----| ----|----|
|<center>GPU|<center>91.63|<center>90.94|<center>337.50|<center>2308.89| 
|<center>NPU|<center>90.85|<center>90.04|<center>94.54|<center>554.09|
|<center>baseline(TinyBERT<sub>4</sub>)|<center>92.6|<center>None|<center>None|<center>None|
|<center>requirement|<center>87.6|<center>None|<center>None|<center>None|



# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md
