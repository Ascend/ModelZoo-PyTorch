# Copyright 2022 Huawei Technologies Co., Ltd
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
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import yaml

from frames_dataset import FramesDataset


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def pre_processing():
    with open(opt.config) as f:
        config = yaml.safe_load(f)

    dataset = FramesDataset(is_train=False, **config['dataset_params'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    out_dir = opt.out_dir
    mkdir(out_dir)

    source_dir = os.path.join(out_dir, "source")
    driving_dir = os.path.join(out_dir, "driving")
    mkdir(source_dir)
    mkdir(driving_dir)

    print("pre processing...")
    num_file = []
    cnt  = 0
    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        x_video = x['video']
        num = x_video.shape[2]
        num_file.append(num)
        source = x_video[:, :, 0].numpy()
        for frame_idx in range(x['video'].shape[2]):
            driving = x_video[:, :, frame_idx].numpy()
            if opt.data_type == "npy":
                file_name = str(cnt) + ".npy"
                source_save_path = os.path.join(source_dir, file_name)
                driving_save_path = os.path.join(driving_dir, file_name)
                np.save(source_save_path, source)
                np.save(driving_save_path, driving)
                cnt += 1
            else:
                file_name = str(cnt) + ".bin"
                source_save_path = os.path.join(source_dir, file_name)
                driving_save_path = os.path.join(driving_dir, file_name)
                source.tofile(f'{source_save_path}')
                driving.tofile(f'{driving_save_path}')
                cnt += 1
    num_file = np.array(num_file)
    np.save(out_dir + "frame_num.npy", num_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--out_dir", default="pre_data/", help="path to checkpoint to restore")
    parser.add_argument("--data_type", default="npy", help="out put file type", choices=["npy", "bin"])
    opt = parser.parse_args()

    pre_processing()
