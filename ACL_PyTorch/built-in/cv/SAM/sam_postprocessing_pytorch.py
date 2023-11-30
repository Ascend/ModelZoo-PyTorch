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


import torch
import numpy as np
from torch.nn import functional as F


postprocessing_config = {
    'image_size': 1024,
    'mode': "bilinear",
    'mask_threshold': 0.0,
}


def resize_longest_image_size(input_image_size, longest_side):
    input_image_size = torch.tensor(input_image_size)
    input_image_size = input_image_size.to(torch.float32)
    scale = longest_side / torch.max(input_image_size)
    transformed_size = scale * input_image_size
    transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
    return transformed_size


def mask_postprocessing(masks, orig_im_size):
    img_size = postprocessing_config['image_size']
    masks = torch.from_numpy(masks)
    masks = F.interpolate(
        masks,
        size=(img_size, img_size),
        mode=postprocessing_config['mode'],
        align_corners=False,
    )

    prepadded_size = resize_longest_image_size(orig_im_size, img_size).to(torch.int64)
    masks = masks[:, :, : prepadded_size[0], :]
    masks = masks[:, :, :, : prepadded_size[1]]

    orig_im_size = torch.tensor(orig_im_size)
    orig_im_size = orig_im_size.to(torch.int64)
    h, w = orig_im_size[0], orig_im_size[1]
    masks = F.interpolate(
        masks, 
        size=(h, w), 
        mode=postprocessing_config['mode'], 
        align_corners=False,
    )
    return masks


def sam_postprocessing(masks, image):
    orig_im_size = np.array(image.shape[:2]).astype(np.float32).tolist()
    upscaled_masks = mask_postprocessing(masks, orig_im_size).numpy()
    upscaled_masks = upscaled_masks > postprocessing_config['mask_threshold']
    return upscaled_masks


