# ENet 训练
This implements training of ENet on the Cityscapes dataset.
- Reference implementation：
```
url=https://github.com/Tramac/awesome-semantic-segmentation-pytorch
```



## Requirements # 

- Install Packages 
- `pip install -r requirements.txt`
   Note: pillow recommends installing a newer version. If the corresponding torchvision version cannot be installed directly, you can use the source code to install the corresponding version. The source code reference link: https://github.com/pytorch/vision，
Suggestion the pillow is 9.1.0 and the torchvision is 0.6.0
- The Cityscapes dataset can be downloaded from the [link](https://www.cityscapes-dataset.com/). 
- Move the datasets to root directory and run the script `unzip.sh`. 
  - `bash ./unzip.sh`



## Training # 
To train a model, change the working directory to `./NPU`,then run: 

```bash
# 1p train perf
bash ./test/train_performance_1p.sh '[your_dataset_path]'

# 8p train perf
bash ./test/train_performance_8p.sh '[your_dataset_path]'

# 1p train full
bash ./test/train_full_1p.sh '[your_dataset_path]'

# 8p train full
bash ./test/train_full_8p.sh '[your_dataset_path]'

# finetuning
bash ./test/train_finetune_1p.sh '[your_dataset_path]'
```
After running,you can see the results in `./NPU/stargan_full_8p/samples` or `./NPU/stargan_full_1p/samples`




## GAN training result # 

|  Type  |   FPS   | Epochs | AMP_Type |
| :----: | :-----: | :----: | :------: |
| NPU-1p | 14.398  |  400   |    O2    |
| NPU-8p | 74.310  |  400   |    O2    |
| GPU-1p | 21.885  |  400   |    O2    |
| GPU-8p | 161.495 |  400   |    O2    |


# Statement
For details about the public address of the code in this repository, you can get from the file public_address_statement.md
