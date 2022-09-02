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
import time


def infer(config, generator, kp_detector, checkpoint, log_dir, dataset):

    data_num = dataset.__len__()
    print("data num is ", data_num)

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

    loss_list = []
    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.cuda()
    kp_detector.cuda()

    generator.eval()
    kp_detector.eval()

    kp_infer_time = 0
    kp_infer_num = 0
    gen_infer_time = 0
    gen_infer_num = 0

    for it, x in tqdm(enumerate(dataloader)):

        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break

        with torch.no_grad():
            x['video'] = x['video'].cuda()
            kp_start = time.perf_counter_ns()
            kp_source = kp_detector(x['video'][:, :, 0])
            kp_end = time.perf_counter_ns()
            kp_infer_time += (kp_end - kp_start)
            kp_infer_num += 1

            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]
                driving = x['video'][:, :, frame_idx]
                kp_start = time.perf_counter_ns()
                kp_driving = kp_detector(driving)
                kp_end = time.perf_counter_ns()
                kp_infer_time += (kp_end - kp_start)
                kp_infer_num += 1

                gen_start = time.perf_counter_ns()
                out = generator(source, kp_source=kp_source, kp_driving=kp_driving)

                gen_end = time.perf_counter_ns()
                gen_infer_time += (gen_end - gen_start)
                gen_infer_num += 1
                del out['sparse_deformed']
                loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())

    kpat = kp_infer_time / kp_infer_num / 1000 / 1000
    genat = gen_infer_time / gen_infer_num / 1000 / 1000

    kpxn = 1000 / kpat
    genxn = 1000 / genat
    print("Reconstruction loss: %s" % np.mean(loss_list))
    kp_msg = 'kp total infer num: ' + str(kp_infer_num) + '\n' + \
             'kp total infer time(ms): ' + str(kp_infer_time / 1000 / 1000) + '\n' + \
             'kp average infer time(ms): ' + str(kpat) + '\n' + \
             'kp 性能为:' + str(kpxn) + '\n'
    print(kp_msg)
    gen_msg = 'gen total infer num: ' + str(gen_infer_num) + '\n' + \
              'gen total infer time(ms): ' + str(gen_infer_time / 1000 / 1000) + '\n' + \
              'gen average infer time(ms): ' + str(genat) + '\n' + \
              'gen 性能为:' + str(genxn) + '\n'
    print(gen_msg)