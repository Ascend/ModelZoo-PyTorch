# HigherHRNet

This implements training of HigherHRNet on the COCO dataset, mainly modified from GitHub - HRNet/HigherHRNet-Human-Pose-Estimation

1. Install package dependencies. Make sure the python environment >=3.7

   ```bash
   pip install -r requirements.txt
   ```
2. Install COCOAPI:

```
# COCOAPI=/path/to/clone/cocoapi

git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI

# Install into global site-packages

make install

# Alternatively, if you do not have permissions or prefer

# not to install the COCO API into global site-packages

python3 setup.py install --user
Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
```

 

3. Download pretrained models from the releases of HigherHRNet-Human-Pose-Estimation to the specified directory

   ```txt
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   `-- hrnet_w32-36af842e.pth
            `-- pose_coco
                `-- pose_higher_hrnet_w32_512.pth
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

```

## HigherHRNet training result

|  名称  | 精度 |  性能 |
| :----: | :--: |  :------: |
| NPU-8p |  66.9 |   2.2s/step   |
| GPU-8p | 67.1  |    1.2s/step    |
| NPU-1p |       |     1.1s/step   |
| GPU-1p |       |    0.7s/step|


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md