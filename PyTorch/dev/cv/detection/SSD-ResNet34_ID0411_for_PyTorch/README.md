# MLPerf_SSD-R34-Large
SSD on Large images with a backbone of ResNet34 based on MLPerf-training single-stage-detector repo 
##Installation
To install the environment please follow the instruction on [MLPerf-training single-stage-detector](https://github.com/mlperf/training/tree/master/single_stage_detector). The files in this repo replace the files in the ssd folder.
#### Please note that the latest version of this repo reuquires [NVIDIA apex](https://github.com/NVIDIA/apex) as well. 

## Changes from original repo:
1. Support training on any data size including images with uneven ratio e.g 1600x1200
2. Support training on multi GPUs
3. Support different strides from command line - this is a list of 6 numbers: default [1,1,2,2,2,1]. The idea is to control the number of anchors 
3. Removed hard coded steps\feature maps sizes 

## Experiments:
1. To train any model you need to specifiy the image size  the batch size (so it ould fit into your memory) and the strides which define the number of anchors. For instance 
   ```
   python train.py --device-ids 0 1 2 3 4 5 6 7 --strides 2 2 2 2 1 1 --batch-size 32 --image-size 800 1200
   ```
2. The training of the final [MLPerf inference V0.5 ResNet34-SSD model](https://zenodo.org/record/3236545#.XS4ibOhKiUk) was done in three stages where in each step we increased the image-size and loaded resumed from previous checkpoint (probably this regime is suboptimal). To reproduce the results please run the follwing lines: 
   ```
   python train.py --device-ids 0 --image-size 300 300 --save-path models_300
   python train.py --device-ids 0 1 2 3 --image-size 700 700 --save-path models_700 --checkpoint ./models_300/iter_24000
   python train.py --device-ids 0 1 2 3 4 5 6 7 --image-size 1200 1200 --strides 3 3 2 2 2 2 --batch-size 32 --save-path models_1200 --checkpoint ./models_700/iter_24000
   ```   
  
Where ```--checkpoint``` flag is used to resume from pre-trained checkpoint.

# Updates
2/20/2019 added SyncBatchNorm from NVIDIA apex and fixed small bugs to make it run on pyt-10 as well
