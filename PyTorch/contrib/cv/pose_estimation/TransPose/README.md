# TransPose

This implements training of TransPose on the COCO dataset, mainly modified from GitHub - yangsenius/TransPose

1. Install package dependencies. Make sure the python environment >=3.7

   ```bash
   pip install -r requirements.txt
   ```

2. Download pretrained models from the releases of GitHub - yangsenius/TransPose to the specified directory

   ```txt
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   |-- hrnet_w32-36af842e.pth
            |   |-- hrnet_w48-8ef0771d.pth
            |   |-- resnet50-19c8e357.pth
            |-- transpose_coco
            |   |-- tp_r_256x192_enc3_d256_h1024_mh8.pth
            |   |-- tp_r_256x192_enc4_d256_h1024_mh8.pth
            |   |-- tp_h_32_256x192_enc4_d64_h128_mh1.pth
            |   |-- tp_h_48_256x192_enc4_d96_h192_mh1.pth
            |   |-- tp_h_48_256x192_enc6_d96_h192_mh1.pth    
   ```

### Data Preparation

Please download or link COCO to ${POSE_ROOT}/data/coco/, and make them look like this:

```txt
${POSE_ROOT}/data/coco/
|-- annotations
|   |-- person_keypoints_train2017.json
|   `-- person_keypoints_val2017.json
|-- person_detection_results
|   |-- COCO_val2017_detections_AP_H_56_person.json
|   `-- COCO_test-dev2017_detections_AP_H_609_person.json
`-- images
	|-- train2017
	|   |-- 000000000009.jpg
	|   |-- ... 
	`-- val2017
		|-- 000000000139.jpg
		|-- ... 
```
## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
# training 1p accuracy
bash ./test/train_full_1p.sh --data_path=real_data_path

# training 1p performance
bash ./test/train_performance_1p.sh --data_path=real_data_path

# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path

#test 8p accuracy
bash test/train_eval_8p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path

# finetuning 1p 
bash test/train_finetune_1p.sh --data_path=real_data_path --pth_path=real_pre_train_model_path
```

## TransPose training result

|  名称  | 精度 | 性能 | AMP_Type |
| :----: | :--: | :--: | :------: |
| GPU-1p |  -   | 0.34s/step  |    O1    |
| GPU-8p | 71.7 | 0.98s/step  |    O1    |
| NPU-1p |  -   | 0.34s/step  |    O1    |
| NPU-8p | 72.5 | 0.95s/step  |    O1    |

