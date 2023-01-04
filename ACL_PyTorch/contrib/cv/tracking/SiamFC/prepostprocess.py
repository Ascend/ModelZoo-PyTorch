# Copyright 2021 Huawei Technologies Co., Ltd
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
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from utils import ToTensor, get_exemplar_image, get_pyramid_instance_image
import struct


exemplar_size = 127                    # exemplar size   z
instance_size = 255                    # instance size   x
context_amount = 0.5                   # context amount
num_scale = 3                          # number of scales
scale_step = 1.0375                    # scale step of instance image
scale_penalty = 0.9745                 # scale penalty
scale_lr = 0.59                        # scale learning rate
response_up_stride = 16                # response upsample stride
response_sz = 17                       # response size
window_influence = 0.176               # window influence
total_stride = 8                       # total stride of backbone


class PrePostProcess(object):
    def __init__(self):
        self.penalty = np.ones((num_scale)) * scale_penalty
        self.penalty[num_scale // 2] = 1  # [0.9745, 1, 0.9745]

        # create cosine window    upsample stride=2^4=16, heatmap 17x17
        self.interp_response_sz = response_up_stride * response_sz  # 272=16x17
        self.cosine_window = self._cosine_window((self.interp_response_sz, self.interp_response_sz))

    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def cropexemplar(self, frame, box, save_path, file_name):
        """
            Args:
                frame: an RGB image
                box: one-based bounding box [x, y, width, height]
        """
        self.bbox = (box[0] - 1, box[1] - 1, box[0] - 1 + box[2], box[1] - 1 + box[3])  # zero based x1，y1,x2,y2
        self.pos = np.array([box[0] - 1 + (box[2]) / 2, box[1] - 1 + (box[3]) / 2])  # zero based cx, cy,
        self.target_sz = np.array([box[2], box[3]])  # zero based w, h

        # get exemplar img
        self.img_mean = tuple(map(int, frame.mean(axis=(0, 1))))
        exemplar_img, scale_z, s_z = get_exemplar_image(frame, self.bbox,
                                                        exemplar_size, context_amount, self.img_mean)

        # create scales: 0.96, 1, 1.037
        self.scales = scale_step ** np.arange(np.ceil(num_scale / 2) - num_scale,
                                              np.floor(num_scale / 2) + 1)

        # create s_x : instance is twice as large as exemplar
        self.s_x = s_z + (instance_size - exemplar_size) / scale_z  # s-x search_sz, s-z exemplar_sz

        # arbitrary scale saturation
        self.min_s_x = 0.2 * self.s_x
        self.max_s_x = 5 * self.s_x

        # get exemplar feature
        # m1: use torchvision.transforms
        # exemplar_img = self.transforms(exemplar_img)[None, :, :, :]  # 1,3,127,127
        # m2: don't use torchvision.transforms
        exemplar_img = ToTensor(exemplar_img)
        img = np.array(exemplar_img).astype(np.float32)
        path = os.path.join(save_path, file_name.split('.')[0].replace('/', '-') + ".bin")
        img.tofile(path)
        return path

    def cropsearch(self, frame, save_path, file_name):
        size_x_scales = self.s_x * self.scales  # multi-scale search
        pyramid = get_pyramid_instance_image(frame, self.pos, instance_size, size_x_scales, self.img_mean)
        # m1: use torchvision.transforms
        # instance_imgs = torch.cat([self.transforms(x)[None, :, :, :] for x in pyramid], dim=0)  # 3, 3, 255, 255
        # m2: don't use torchvision.transforms
        instance_imgs = torch.cat([ToTensor(x) for x in pyramid], dim=1)  # 3, 3, 255, 255
        img = np.array(instance_imgs).astype(np.float32)
        path = os.path.join(save_path, file_name.split('.')[0].replace('/', '-') + ".bin")
        img.tofile(path)
        return path

    def postprocess(self, x_f, z_f):
        # x_f:search  z_f:exemplar
        response_maps = F.conv2d(x_f, z_f, groups=3)
        response_maps = response_maps.transpose(0, 1)
        response_maps = response_maps.numpy().squeeze()  # 3, 17, 17

        response_maps_up = [cv2.resize(x, (self.interp_response_sz, self.interp_response_sz), cv2.INTER_CUBIC)
                            for x in response_maps]  # upsample

        # get max score of each scale
        max_score = np.array([x.max() for x in response_maps_up]) * self.penalty  # penalty=[0.9745, 1, 0.9745]

        # penalty scale change
        scale_idx = max_score.argmax()
        response_map = response_maps_up[scale_idx]
        response_map -= response_map.min()
        response_map /= response_map.sum()
        response_map = (1 - window_influence) * response_map + \
                       window_influence * self.cosine_window
        max_r, max_c = np.unravel_index(response_map.argmax(), response_map.shape)
        # displacement in interpolation response
        disp_response_interp = np.array([max_c, max_r]) - (self.interp_response_sz - 1) / 2.
        # displacement in input, response_up_stride=16, total_stride=8
        disp_response_input = disp_response_interp * total_stride / response_up_stride
        # displacement in frame
        scale = self.scales[scale_idx]  #
        disp_response_frame = disp_response_input * (self.s_x * scale) / instance_size
        # position in frame coordinates
        self.pos += disp_response_frame
        # scale damping and saturation
        self.s_x *= ((1 - scale_lr) + scale_lr * scale)  # update
        self.s_x = max(self.min_s_x, min(self.max_s_x, self.s_x))
        self.target_sz = ((1 - scale_lr) + scale_lr * scale) * self.target_sz  # update

        box = np.array([
            self.pos[0] + 1 - (self.target_sz[0]) / 2,
            self.pos[1] + 1 - (self.target_sz[1]) / 2,
            self.target_sz[0], self.target_sz[1]])

        return box

    def file2tensor(self, filepath, shape):
        size = os.path.getsize(filepath)
        res = []
        L = int(size / 4)  # float32, so 4bytes
        binfile = open(filepath, 'rb')
        for i in range(L):
            data = binfile.read(4)
            num = struct.unpack('f', data)
            res.append(num[0])
        binfile.close()

        dim_res = np.array(res).reshape(shape)
        tensor_res = torch.tensor(dim_res, dtype=torch.float32)

        return tensor_res
