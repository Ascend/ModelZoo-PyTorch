# StarGAN 训练
This implements training of StarGAN on the CelebA dataset.
- Reference implementation：
```
url=https://github.com/yunjey/stargan
```



## Requirements # 

- Install Packages 
- `pip install -r requirements.txt`
- The CelebA dataset can be downloaded from the [link](https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0). You can use `wget` to download as well. 
  - `wget -N https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0`
- Move the datasets to root directory and run the script `unzip_dataset.sh`. 
  - `bash ./unzip_dataset.sh`



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

| Type | FPS       | Epochs   | AMP_Type |
| :------: | :------:  | :------: | :------: |
| NPU-1p   | 95     | 1      | O1       |
| NPU-8p | 615   | 50      | O1       |
| GPU-1p | 62 | 1 | O1 |
| GPU-8p | 517 | 50 | O1 |



