# Preparing UCF-101

## Introduction
```
@article{Soomro2012UCF101AD,
  title={UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild},
  author={K. Soomro and A. Zamir and M. Shah},
  journal={ArXiv},
  year={2012},
  volume={abs/1212.0402}
}
```
For basic dataset information, you can refer to the dataset [website](https://www.crcv.ucf.edu/research/data-sets/ucf101/).

Before we start, please make sure that you have set environment constants. If not, you can run the following script.
```
source ../test/env.sh
```

## Step 1. Prepare Annotations
First of all, you can run the following script to prepare annotations.
```
bash download_annotations.sh
```
The dataset will be saved in {current_dir}/ucf101

## Step 2. Prepare Videos
Then, you can run the following script to prepare videos.
```
bash download_videos.sh
```

## Step 3. Extract RGB
You can still extract RGB frames using OpenCV by the following script, but it will keep the original size of the images.
```
bash extract_rgb_frames_opencv.sh
```

## Step 4. Generate File List
You can run the follow script to generate file list in the format of rawframes and videos.
```
bash generate_videos_filelist.sh
bash generate_rawframes_filelist.sh
```

## Step 5. Move Dataset (Optional)
In this project, we save dataset in directory /opt/npu/ with the follow command
```
mv ucf101 /opt/npu/
```
If you save dataset in other path, please modify dataset path in ../config/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb.py.

## Step 6. Check Directory Structure
After the whole data process for UCF-101 preparation, you will get the rawframes, videos and annotation files for UCF-101.

In the context of the whole project, the folder structure will look like:
```
opt
├── npu
│   ├── ucf101
│   │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
│   │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
│   │   ├── annotations
│   │   ├── videos
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi

│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g25_c05.avi
│   │   ├── rawframes
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01
│   │   │   │   │   ├── img_00001.jpg
│   │   │   │   │   ├── img_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g01_c01
│   │   │   │   ├── ...
│   │   │   │   ├── v_YoYo_g25_c05
```
