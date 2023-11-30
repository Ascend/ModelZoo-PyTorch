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


import cv2
import torch
import numpy as np
from torch.nn import functional as F
from segment_anything.utils.transforms import ResizeLongestSide


preprocessing_config = {
    'image_size': 1024,
    'mean':[123.675, 116.28, 103.53],
    'std':[58.395, 57.12, 57.375],
}


def encoder_preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = ResizeLongestSide(preprocessing_config['image_size'])
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device='cpu')
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    pixel_mean = torch.Tensor(preprocessing_config['mean']).view(-1, 1, 1)
    pixel_std = torch.Tensor(preprocessing_config['std']).view(-1, 1, 1)
    x = (input_image_torch - pixel_mean) / pixel_std
    h, w = x.shape[-2: ]
    padh = preprocessing_config['image_size'] - h
    padw = preprocessing_config['image_size'] - w
    x = F.pad(x, (0, padw, 0, padh))
    x = x.numpy().astype(np.float32)
    return x


def decoder_preprocessing(image_embedding, input_point, image):
    input_point = np.array(input_point)
    input_label = [1] * len(input_point)
    input_label = np.array(input_label)
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    transform = ResizeLongestSide(preprocessing_config['image_size'])
    onnx_coord = transform.apply_coords(onnx_coord, image.shape[: 2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    decoder_inputs = [image_embedding, onnx_coord, onnx_label, onnx_mask_input, onnx_has_mask_input]
    return decoder_inputs
