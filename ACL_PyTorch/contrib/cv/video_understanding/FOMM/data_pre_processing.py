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
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger
import numpy as np
from sync_batchnorm import DataParallelWithCallback
from my_utils import mkdir

def pre_processing(config, generator, kp_detector, checkpoint, log_dir, dataset, data_type, out_dir="pre_data/"):

    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    if not out_dir.__contains__("/"):
        out_dir += "/"

    mkdir(out_dir)

    source_dir = out_dir + "source/"
    driving_dir = out_dir + "driving/"
    mkdir(source_dir)
    mkdir(driving_dir)

    num_file = []
    cnt  = 0
    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            x_video = x['video']
            num = x_video.shape[2]
            num_file.append(num)
            source = x_video[:, :, 0].numpy()
            source_path = source_dir
            driving_path = driving_dir
            mkdir(source_path)
            mkdir(driving_path)
            for frame_idx in range(x['video'].shape[2]):
                if data_type == "npy":
                    file_name = str(cnt) + ".npy"
                    driving = x_video[:, :, frame_idx].numpy()
                    source_save_path = source_path + file_name
                    driving_save_path = driving_path + file_name

                    np.save(source_save_path, source)
                    np.save(driving_save_path, driving)
                    cnt += 1
                else:
                    file_name = str(cnt) + ".bin"
                    driving = x_video[:, :, frame_idx].numpy()
                    source_save_path = source_path + file_name
                    driving_save_path = driving_path + file_name
                    source.tofile(f'{source_save_path}')
                    driving.tofile(f'{driving_save_path}')
                    cnt += 1
    num_file = np.array(num_file)
    np.save(out_dir + "frame_num.npy", num_file)