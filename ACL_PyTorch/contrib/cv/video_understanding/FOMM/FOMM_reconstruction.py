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
from shutil import copy

import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
import imageio
from tqdm import tqdm

from frames_dataset import FramesDataset


def reconstruction():
    with open(opt.config) as f:
        config = yaml.safe_load(f)

    pre_data = opt.pre_data
    data_dir = opt.data_dir
    png_dir = opt.png_dir

    dataset = FramesDataset(is_train=False, **config['dataset_params'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if not data_dir[-1] == '/':
        data_dir += "/"

    if not pre_data[-1] == '/':
        pre_data += "/"
        
    kpdv_path = data_dir + "kpdv/"
    kpdj_path = data_dir + "kpdj/"
    kpsv_path = data_dir + "kpsv/"
    kpsj_path = data_dir + "kpsj/"
    source_path = pre_data + "source/"
    driving_path = pre_data + "driving/"
    out_path = data_dir + "out/"

    cnt = 0
    print("Reconstruction...")

    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        num = x['video'].shape[2]
        del x['video']
        file_num_file = np.load(pre_data + "frame_num.npy")
        file_num = file_num_file[it]
        if num != file_num:
            raise ValueError("{}:file num != num, num is {}, but file num is {}".format(it, num, file_num))
        predictions = []
        visualizations = []
        for i in range(num):
            out = dict()
            kp_driving = dict()
            kp_source = dict()
            for j in range(5):
                if j == 1:
                    continue
                outi_path = out_path + str(cnt) + "_" + str(j) + ".npy"
                outi = np.load(outi_path)
                if j == 0:
                    out['mask'] = torch.from_numpy(outi).to(torch.float64)
                elif j == 2:
                    out['occlusion_map'] = torch.from_numpy(outi).to(torch.float64)
                elif j == 3:
                    out['deformed'] = torch.from_numpy(outi).to(torch.float64)
                elif j == 4:
                    out['prediction'] = torch.from_numpy(outi).to(torch.float64)

            kp_driving_value_name = kpdv_path + str(cnt) + ".npy"
            kp_driving_jac_name = kpdj_path + str(cnt) + ".npy"
            kp_source_value_name = kpsv_path + str(cnt) + ".npy"
            kp_source_jac_name = kpsj_path + str(cnt) + ".npy"
            source_name = source_path + str(cnt) + ".npy"
            driving_name = driving_path + str(cnt) + ".npy"

            kp_driving_value = np.load(kp_driving_value_name)
            kp_driving_jac = np.load(kp_driving_jac_name)
            kp_source_value = np.load(kp_source_value_name)
            kp_source_jac = np.load(kp_source_jac_name)
            source = np.load(source_name)
            driving = np.load(driving_name)

            cnt += 1

            kp_driving['value'] = torch.from_numpy(kp_driving_value).to(torch.float64)
            kp_driving['jacobian'] = torch.from_numpy(kp_driving_jac).to(torch.float64)
            kp_source['value'] = torch.from_numpy(kp_source_value).to(torch.float64)
            kp_source['jacobian'] = torch.from_numpy(kp_source_jac).to(torch.float64)

            out['kp_source'] = kp_source
            out['kp_driving'] = kp_driving

            source = torch.from_numpy(source).to(torch.float64)
            driving = torch.from_numpy(driving).to(torch.float64)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

        predictions = np.concatenate(predictions, axis=1)
        imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--png_dir", default='checkpoint/reconstruction/png', help="path to png")
    parser.add_argument("--data_dir", default="infer_out/", help="root path of infer output")
    parser.add_argument("--pre_data", default="pre_data/", help="path to data preprocessed")

    opt = parser.parse_args()

    reconstruction()
