# ReID Strong Baseline

## Get Started
1. Install dependencies:

    - [pytorch>=0.4]
    - torchvision
    - [pytorch-ignite=0.1.2](https://github.com/pytorch/ignite) (Note: V0.2.0 may result in an error)
    - [yacs](https://github.com/rbgirshick/yacs)

2. Prepare dataset

    （1）Market1501

    * Download the training and validation set of [Market1501](https://github.com/michuanhaohao/reid-strong-baseline) 
    * Run `unzip Market-1501-v15.09.15.zip  ` to  unzip the dataset and rename to `market1501`. The data structure would like:
    
    ```bash
    data
        market1501 # this folder contains 6 files.
            bounding_box_test/
            bounding_box_train/
            ......
    ```
    and then you should set the path in all the  file in `./test/*.sh` about `DATASETS.ROOT_DIR`

3. Prepare pretrained model if you don't have
   
    （1）ResNet
    
    * Download pretrained model [resnet50-19c8e357.pth](https://ascend-pytorch-one-datasets.obs.cn-north-4.myhuaweicloud.com/train/pth/resnet50-19c8e357.pth)
    
    * `mkdir $ReidStrongBaseline/.cache` 
   
    * and then you put it in the path to be  `./.cache/*.pth` 
    

    
4. Modify the function of `apex` package(If you use the `apex` package after 2021081000,  you can ignore it )
    add some code in `${PYTHONPATH}/python3.7/site-packages/apex/amp/scaler.py line:307`

    ```bash
    if master_grads_combined is None:
        return
    ```

## Train
1. If run the model on the Linux system，you should run the code to convert the format:

   ```bash
   dos2unix test/*.sh
   ```

2. Market1501, cross entropy loss + triplet loss

```bash
cd $ReidStrongBaseline

# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=xxx 
#$data_path for real path to Market1501_datasets

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=xxx 
#$data_path for real path to Market1501_datasets

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=xxx 
#$data_path for real path to Market1501_datasets

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=xxx 
#$data_path for real path to Market1501_datasets
```

2. show the prof_demo and inference .


```bash
sh demo.sh
sh prof_demo.sh
```

## Test
You can test your model's performance directly by running these commands in `.sh ` files after your custom modification. 

```bash
# evaluation 1p accuracy
bash ./test/eval_1p.sh --data_path=xxx 
#$data_path for real path to Market1501_datasets

# evaluation 8p accuracy
bash ./test/eval_8p.sh --data_path=xxx 
#$data_path for real path to Market1501_datasets
```

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md