# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time
import warnings
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm
from .alexnet import SiameseAlexNet
from .config import config
from .custom_transforms import ToTensor
from .utils import get_exemplar_image, get_pyramid_instance_image, get_instance_image, show_image

torch.set_num_threads(1)  # otherwise PyTorch will take all cpus


class SiamFCTracker:
    def __init__(self, model_path, npu_id=0, is_deterministic=False):
        
        self.npu_id = npu_id
        self.name = 'SiamFC'
        self.is_deterministic = is_deterministic

        self.model = SiameseAlexNet()
        self.model.load_state_dict(torch.load(model_path))
        torch.npu.set_device(npu_id)
        self.model = self.model.to(torch.device("npu:{}".format(self.npu_id)))
        self.model.eval()

        self.transforms = transforms.Compose([ToTensor()])

    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, box):
        """ initialize siamfc tracker
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        self.bbox = (box[0]-1, box[1]-1, box[0]-1+box[2], box[1]-1+box[3])  # zero based x1, y1, x2, y2
        self.pos = np.array([box[0]-1+(box[2])/2, box[1]-1+(box[3])/2])     # zero based cx, cy
        self.target_sz = np.array([box[2], box[3]])                         # zero based w, h

        # get exemplar img
        self.img_mean = tuple(map(int, frame.mean(axis=(0, 1))))
        exemplar_img, scale_z, s_z = get_exemplar_image(frame, self.bbox,
                                                        config.exemplar_size, config.context_amount, self.img_mean)

        # get exemplar feature
        exemplar_img = self.transforms(exemplar_img)[None, :, :, :]  # 1, 3, 127, 127

        exemplar_img_var = Variable(exemplar_img.to(torch.device("npu:{}".format(self.npu_id))))
        self.model((exemplar_img_var, None))

        # penalty designed in update
        self.penalty = np.ones(config.num_scale) * config.scale_penalty
        self.penalty[config.num_scale//2] = 1  # [0.9745, 1, 0.9745]

        # create cosine window, stride of upsample=2^4=16, heatmap size=17x17
        self.interp_response_sz = config.response_up_stride * config.response_sz  # 272=16x17
        self.cosine_window = self._cosine_window((self.interp_response_sz, self.interp_response_sz))
        # create scales ratio: 0.96, 1, 1.037
        self.scales = config.scale_step ** np.arange(np.ceil(config.num_scale/2)-config.num_scale,
                                                     np.floor(config.num_scale/2)+1)
        
        # create s_x, instance is twice as large as exemplar
        self.s_x = s_z + (config.instance_size-config.exemplar_size) / scale_z  # s-x: search, s-z: exemplar

        # arbitrary scale saturation
        self.min_s_x = 0.2 * self.s_x

        self.max_s_x = 5 * self.s_x

    def update(self, frame):
        """track object based on the previous frame
        Args:
            frame: an RGB image
        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        size_x_scales = self.s_x * self.scales  # multi-scale
        pyramid = get_pyramid_instance_image(frame, self.pos, config.instance_size, size_x_scales, self.img_mean)
        instance_imgs = torch.cat([self.transforms(x)[None, :, :, :] for x in pyramid], dim=0)  # [3, 3, 255, 255]

        instance_imgs_var = Variable(instance_imgs.to(torch.device("npu:{}".format(self.npu_id))))
        response_maps = self.model((None, instance_imgs_var))  # 3, 1, 17, 17
        response_maps = response_maps.data.cpu().numpy().squeeze()  # 3, 17, 17
        response_maps_up = [cv2.resize(x, (self.interp_response_sz, self.interp_response_sz), cv2.INTER_CUBIC)
                            for x in response_maps]  # upsample the 17*17 response

        # get max score of each scale
        max_score = np.array([x.max() for x in response_maps_up]) * self.penalty  # penalty=[0.9745, 1, 0.9745]
        
        # penalty scale change
        scale_idx = max_score.argmax()  # get the scale which has the max score
        response_map = response_maps_up[scale_idx]  # get response map of the scale
        response_map -= response_map.min()  # minus minimum
        response_map /= response_map.sum()  # normalize
        response_map = (1 - config.window_influence) * response_map + config.window_influence * self.cosine_window
        max_r, max_c = np.unravel_index(response_map.argmax(), response_map.shape)  # location of idx in response-map
        # displacement in interpolation response, according to the center
        disp_response_interp = np.array([max_c, max_r]) - (self.interp_response_sz-1) / 2.
        # displacement in input, response_up_stride=16, total_stride=8
        disp_response_input = disp_response_interp * config.total_stride / config.response_up_stride
        # displacement in frame
        scale = self.scales[scale_idx]  #
        disp_response_frame = disp_response_input * (self.s_x * scale) / config.instance_size  # true displacement
        # position in frame coordinates
        self.pos += disp_response_frame
        # scale damping and saturation
        self.s_x *= ((1 - config.scale_lr) + config.scale_lr * scale)  # update size
        self.s_x = max(self.min_s_x, min(self.max_s_x, self.s_x))
        self.target_sz = ((1 - config.scale_lr) + config.scale_lr * scale) * self.target_sz  # update size

        box = np.array([
           self.pos[0] + 1 - (self.target_sz[0]) / 2,
           self.pos[1] + 1 - (self.target_sz[1]) / 2,
           self.target_sz[0], self.target_sz[1]])

        return box

    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box  # x, y, w, h
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            begin = time.time()
            if f == 0:  # first frame
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                show_image(img, boxes[f, :])

        return boxes, times
