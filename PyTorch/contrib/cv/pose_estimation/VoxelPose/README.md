# VoxelPose

This implements training of voxelpose on the MultiHumanPose's Shelf Dataset, mainly modified from [microsoft/voxelpose-pytorch](https://github.com/microsoft/voxelpose-pytorch/).

## VoxelPose Detail

### Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt` and then install the specified version of torch, torchvision and apex
- Download the dataset from http://campar.in.tum.de/Chair/MultiHumanPose (Shelf Dataset) and extract them under ${POSE_ROOT}/data/Shelf
- Put the [panoptic_training_pose.pkl](https://github.com/microsoft/voxelpose-pytorch/blob/main/data/panoptic_training_pose.pkl) under ${POSE_ROOT}/data

### Add Shape
Set AscendProject = /usr/local/Ascend #use real path
- Add shape [32, 16, 1, 1, 1], [1, 32, 1, 1, 1], [17, 32, 1, 1, 1] into ${AscendProject}/ascend-toolkit/5.0.2/arm64-linux/opp/op_impl/built-in/ai_core/tbe/impl/fractal_z_3d_2_ncdhw.py
- Add shape [262144, 1, 1, 16, 16], [128000, 1, 1, 16, 16] into ${AscendProject}/ascend-toolkit/5.0.2/arm64-linux/opp/op_impl/built-in/ai_core/tbe/impl/fractal_z_3d_2_ncdhw.py

### Training
To train a model, run the following scripts with the real path of dataset:
PS：There is a large jitter on single card training with bs=1, do not perform single card trainning.

```bash
# real_data_path = data/shelf
# training 8p accuracy
bash ./test/train_full_8p.sh --data_path=real_data_path

# training 8p performance
bash ./test/train_performance_8p.sh --data_path=real_data_path

# test 8p accuracy
bash test/train_eval_8p.sh
```

Log path:
    ${POSE_ROOT}/output/shelf_synthetic/multi_person_posenet_50/prn64_cpn80x80x20/prn64_cpn80x80x20_{time}_train.log  # training detail log



### VoxelPose training result
说明：由于模型单卡训练（bs=1）抖动大，loss不收敛，暂不持支单卡训练。

| 名称    | 精度       | FPS | AMP_Type   |
| :------: | :------:  | :------: | :------: |
|  NPU-8p  |   97.10    |    1.267     |   O1    |


# Statement

For details about the public address of the code in this repository, you can get from the file public_address_statement.md