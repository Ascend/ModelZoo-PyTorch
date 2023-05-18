# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import argparse
 
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
 
from det3d import torchie
from det3d.datasets import build_dataset
from det3d.datasets.kitti.kitti import KittiDataset
from det3d.datasets.kitti.kitti_common import (
    get_kitti_image_info, 
    _read_imageset_file, 
    _calculate_num_points_in_gt,
    _create_reduced_point_cloud,
)
from det3d.torchie.parallel import collate_kitti
 
 
def join_path_and_create_dir(path: str, *args) -> str:
    '''
    Create directory if not exist
    '''
    path = os.path.join(path, *args)
    if not os.path.exists(path):
        os.makedirs(path)
 
    return path
 
 
def create_kitti_info_file(
        data_path: str, 
        save_path=".", 
        _train=False, 
        _val=False, 
        _train_val=False, 
        _test=False, 
        reduced=True
    ) -> None:
    imageset_folder = os.path.join("SE-SSD", "det3d", "datasets", "ImageSets")
 
    print("Generate info. this may take several minutes.")
    relative_path = False
 
    file_names = {
        "train": os.path.join(save_path, "kitti_infos_train.pkl"),
        "val": os.path.join(save_path, 'kitti_infos_val.pkl'),
        "train_val": os.path.join(save_path, 'kitti_infos_trainval.pkl'),
        "test": os.path.join(save_path, 'kitti_infos_test.pkl'),
    }
 
    if _train or _train_val:
        train_img_ids = _read_imageset_file(os.path.join(imageset_folder, "train.txt"))
        kitti_infos_train = get_kitti_image_info(
            data_path,
            training=True,
            label_info=True,
            velodyne=True,
            calib=True,
            image_ids=train_img_ids,
            relative_path=relative_path,
        )
        _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
 
    if _train:
        filename = file_names["train"]
        with os.fdopen(os.open(filename, os.O_RDWR|os.O_CREAT, 0o644), "wb") as f:
            pickle.dump(kitti_infos_train, f)
            print(f"The train info file has been saved to {filename}")
 
    if _val or _train_val:
        val_img_ids = _read_imageset_file(os.path.join(imageset_folder, "val.txt"))
        kitti_infos_val = get_kitti_image_info(data_path,
                                            training=True,
                                            label_info=True,
                                            velodyne=True,
                                            calib=True,
                                            image_ids=val_img_ids,
                                            relative_path=relative_path)
        _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
 
    if _val:
        filename = file_names["val"]
        with os.fdopen(os.open(filename, os.O_RDWR|os.O_CREAT, 0o644), "wb") as f:
            pickle.dump(kitti_infos_val, f)
            print(f"The val info file has been saved to {filename}")
 
    if _train_val:
        filename = file_names["train_val"]
        with os.fdopen(os.open(filename, os.O_RDWR|os.O_CREAT, 0o644), "wb") as f:
            pickle.dump(kitti_infos_train + kitti_infos_val, f)
            print(f"The trainval info file has been saved to {filename}")
 
    if _test:
        test_img_ids = _read_imageset_file(os.path.join(imageset_folder, "test.txt"))
        kitti_infos_test = get_kitti_image_info(data_path,
                                                    training=False,
                                                    label_info=False,
                                                    velodyne=True,
                                                    calib=True,
                                                    image_ids=test_img_ids,
                                                    relative_path=relative_path)
        filename = file_names["test"]
        with os.fdopen(os.open(filename, os.O_RDWR|os.O_CREAT, 0o644), "wb") as f:
            pickle.dump(kitti_infos_test, f)
            print(f"The test info test file is saved to {filename}")
 
    # Generate reduced point cloud data.
    if reduced:
        # check path
        if _train or _val or _train:
            join_path_and_create_dir(data_path, "training", "velodyne_reduced")
 
        if _test:
            join_path_and_create_dir(data_path, "testing", "velodyne_reduced")
 
        print("Generating reduced point cloud samples...")
        if _train or _train_val:
            os.makedirs(data_path)
            _create_reduced_point_cloud(data_path, file_names["train"])
 
        if _val or _train_val:
            _create_reduced_point_cloud(data_path, file_names["val"])
 
        if _test:
            _create_reduced_point_cloud(data_path, file_names["test"])
 
 
def pad_sample(sample: dict, key_list: list, _target_size: int) -> None:
    '''
    IF x < target_size:
        sample["voxels"].shape: (x, 5, 4) -> (target_size, 5, 4)
        sample["coordinates"].shape: (x, 3) -> (target_size, 3)
        sample["num_points"].shape: (x, ) -> (target_size, )
    '''
    for key in key_list:
        if sample[key].shape[0] < _target_size:
            padding_shape = list(sample[key].shape)
            pad = [0 for _ in range(2 * len(padding_shape))]
            pad[-1] = _target_size - padding_shape[0]
            sample[key] = F.pad(sample[key], pad, "constant", 0)
 
    return sample
 
 
def save_dataset_to_bin_files(dataset: KittiDataset, _save_dir: str, _batch_size: int, _target_size: int) -> None:
    save_key_list = ["voxels", "coordinates", "num_points"]
    save_paths = {key: join_path_and_create_dir(_save_dir, key) for key in save_key_list}
 
    data_loader = DataLoader(dataset, batch_size=_batch_size, sampler=None, collate_fn=collate_kitti, shuffle=False,)
 
    n_samples = len(data_loader)
    
    print("Saving binaries...")
    for idx, sample in enumerate(data_loader):
        print(f"[{idx + 1} / {n_samples}]", end="\r")
 
        sample = pad_sample(sample, save_key_list, _target_size * _batch_size)
 
        for key in save_key_list:
            data = sample[key].numpy().astype(np.float32)
            data.tofile(os.path.join(save_paths[key], f"{key}_{idx:05d}.bin"))
        
    print("\nDone.")
 
 
def create_dataset_from_config(subset_config: dict, data_path: str, info_file: str) -> KittiDataset:
    # modify paths
    subset_config.root_path = data_path
    subset_config.info_path = info_file
 
    # build dataset
    dataset = build_dataset(subset_config)
 
    return dataset
 
 
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default='SE-SSD/examples/second/configs/config.py', 
        help="Test config file path"
    )
    parser.add_argument("--data_root", type=str, required=True,  help="Root path of dataset.")
    parser.add_argument("--save_dir", type=str, default='./preprocessed_data',  help="Path to save preprocessed data.")
    parser.add_argument("--info_save_dir", type=str, default='./',  help="Path to save dataset information pkl file.")
    parser.add_argument("--train", action="store_true", help="Create info for train set.")
    parser.add_argument("--val", action="store_true", help="Create info for val set.")
    parser.add_argument("--train_val", action="store_true", help="Create info for train_val set.")
    parser.add_argument("--test", action="store_true", help="Create info for test set.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size of data.")
    return parser.parse_args()
 
 
if __name__ == "__main__":
    settings = parse_arguments()
    info_save_dir = settings.info_save_dir
    data_root = settings.data_root
    batch_size = settings.batch_size
    save_dir = settings.save_dir
 
    train = settings.train
    val = settings.val
    train_val = settings.train_val
    test = settings.test
 
    if not (train or val or train_val or test):
        train = val = train_val = test = True
 
    # build valset info file
    print("Getting dataset information...")
    create_kitti_info_file(data_root, info_save_dir, train, val, train_val, test)
 
    # parse config
    config = torchie.Config.fromfile(settings.config)
    target_size = config.voxel_generator.max_voxel_num
 
    if train:
        print("Processing train data.")
        train_info_file = os.path.join(info_save_dir, 'kitti_infos_train.pkl')
        train_dataset = create_dataset_from_config(config.data.train, data_root, train_info_file)
        save_dataset_to_bin_files(train_dataset, os.path.join(save_dir, "train"), batch_size, target_size)
 
    if val:
        print("Processing val data.")
        val_info_file = os.path.join(info_save_dir, 'kitti_infos_val.pkl')
        config.data.val.test_mode = True
        val_dataset = create_dataset_from_config(config.data.val, data_root, val_info_file)
        save_dataset_to_bin_files(val_dataset, os.path.join(save_dir, "val"), batch_size, target_size)
 
    if train_val:
        print("Processing trainval data.")
        train_val_info_file = os.path.join(info_save_dir, 'kitti_infos_trainval.pkl')
        train_val_dataset = create_dataset_from_config(config.data.trainval, data_root, train_val_info_file)
        save_dataset_to_bin_files(train_val_dataset, os.path.join(save_dir, "trainval"), batch_size, target_size)
 
    if test:
        print("Processing test data.")
        test_info_file = os.path.join(info_save_dir, 'kitti_infos_test.pkl')
        test_dataset = create_dataset_from_config(config.data.test, data_root, test_info_file)
        save_dataset_to_bin_files(test_dataset, os.path.join(save_dir, "test"), batch_size, target_size)
    