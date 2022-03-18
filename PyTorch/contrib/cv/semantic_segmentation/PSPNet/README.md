# PSPNET

This implements training of PSPNet on the PASCAL VOC Aug dataset, mainly modified from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). 

**ps**

1. As of the current date, Ascend-Pytorch doesn't support SyncBN, the backbone uses BN instead. To get a similar performance to SyncBN, we set a larger batch size of 16 rather than 4 in mmsegmentation.
2. Semantic segmentation is trained by iteration. The model trained on 1 NPU is useless, so we do not give the evaluation script for the 1p model.


## Environment  preparation

The latest Ascend-Pytorch version is 1.5.0. MMSegmentation 0.10.0 and mmcv 1.2.7 are chosen as they support pytorch1.5.0.

1. Install the latest Ascend-Pytorch.

2. Downding the repository of Ascned Model Zoo to the folder of `$YOURMODELZOO`

```
# download source code
cd $YOURMODELZOO
git clone https://gitee.com/KevinKe/modelzoo
# go to pspnet
cd contrib/PyTorch/Research/cv/semantic_segmentation/PSPNet
```

Denot `$PSPNET` as the path of `$YOURMODELZOO/contrib/PyTorch/Research/cv/semantic_segmentation/PSPNet`.

3. Build mmcv using

Firstly, download [mmcv1.2.7](https://github.com/open-mmlab/mmcv/tree/v1.2.7) to the path `$YOURMMVCPATH`. Then, copy the `mmcv_replace` to `$YOURMMVCPATH/mmcv`.

Check the numpy version is 1.21.2.

```
# configure
cd $PSPNET
source env_npu.sh

# copy
rm -rf $YOURMMVCPATH/mmcv
mkdir mmcv
cp -r mmcv_replace/* $YOURMMVCPATH/mmcv/

# compile
cd $YOURMMVCPATH
export MMCV_WITH_OPS=1
export MAX_JOBS=8
python3.7.5 setup.py build_ext
python3.7.5 setup.py develop
pip3.7.5 list | grep mmcv
```

Then go back to the $PSPNET folder
```
cd $PSPNET
```

4. Permission configuration
```
chmod -R 777 ./
```

5. remove the `mmcv_replace` folder
```
rm -rf mmcv_replace
```

## Dataset Preparation

1. Download the training and validation set of [PASCAL VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and [PASCAL VOC2010 dataset](https://ascend-test-dataset.obs.cn-north-4.myhuaweicloud.com/train/zip/VOCtrainval_03-May-2010.tar). 

After decompression, the structure of the dataset folder should be:
```none
├── VOCdevkit
│   │   ├── VOC2012
│   │   ├── VOC2010
 ```  

2. Download [PASCALAug dataset](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz).
After depressing, copy `benchmark_REALSE/dataset` to `VOCaug` in the `VOCdevkit` folder.
The structure of the dataset folder should be:
```none
├── VOCdevkit
│   │   ├── VOC2012
│   │   ├── VOC2010
│   │   ├── VOCaug
```
3. Convert the VOCAug dataset using

```
cd $PSPNET
python tools/convert_datasets/voc_aug.py data/VOCdevkit data/VOCdevkit/VOCaug --nproc 8
```

**Note: ** `Segmentation fault (core dumped)` may rise. The reason is that mmcv needs the support of pytorch. Go back to repo folder and run `source env_npu.sh` first. 

4. [Optional] Make a soft link of the dataset to the folder of mmseg100
```
cd $PSPNET
mkdir data
ln -s VOCdevkit data # data_path=./data/VOCdevkit/VOC2012
```

## Training

 **Note [Optional]:** When running scripts, the error `$'\r': command not found` may rise. Use `dos2unix  script_file_name` to change it from window format to Unix format first.


For PSPNet
```bash
cd $PSPNET
source npu_env.sh

# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=xxx 
# --data_path=data/VOCdevkit/VOC2012

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=xxx 

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=xxx 

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=xxx 

# evaluation 8p accuracy
bash ./test/train_val_8p.sh --data_path=xxx 
```
Log and checkpoit path:
```
./output/devie_id/PSPNet/train_${device_id}.log             # training detail log
./output/devie_id/PSPNet/PSPNet_bs16_8p_acc.log             # 8p training performance result log
./output/devie_id/PSPNet/ckpt                               # checkpoits
./output/devie_id/PSPNet_prof/PSPNet_bs16_8p_acc.log        # 8p training accuracy result log
```


### PSPNET with 8p

| device | fps |  aAcc |  mIoU | mAcc |
| :------: | :------: | :------: | :------: | :------: |
|mmsegmentaion| |-- | 76.78| -- |
|GPU-8p| 82.296| 94.92 | 77.13 | 85.7 |
|NPU-8p| 117.808 | 94.71 | 77.04 | 86.52 |

ps: 2x training data are used for training on NPU (`bs*#NPU*#iter=16*8*10000`) v.s. GPU (`bs*#GPU*#iter=4*8*20000`)

