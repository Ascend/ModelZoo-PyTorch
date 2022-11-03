# Copyright 2022 Huawei Technologies Co., Ltd
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

import math
import numpy as np
from preprocess import get_affine_transform, affine_transform


def transform_preds(coords, center, scale, output_size):
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0
    scale_x = scale[0] / output_size[0]
    scale_y = scale[1] / output_size[1]
    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
    return target_coords



def get_final_preds(regression_preds, center, scale, img_size):
    """Get final keypoint predictions from regression vectors and transform
    them back to the image.

    Note:
        batch_size: N
        num_keypoints: K

    Args:
        regression_preds (np.ndarray[N, K, 2]): model prediction.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        img_size (list(img_width, img_height)): model input image size.


    Returns:
        preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    N, K, _ = regression_preds.shape
    preds, maxvals = regression_preds, np.ones((N, K, 1), dtype=np.float32)
    preds = preds * img_size
    # Transform back to the image
    for i in range(N):
        preds[i] = transform_preds(preds[i], center, scale, img_size)
    return preds, maxvals
    

def fliplr_regression(regression,
                      flip_pairs,
                      center_mode='static',
                      center_x=0.5,
                      center_index=0):
    """Flip human joints horizontally.

    Note:
        batch_size: N
        num_keypoint: K
    Args:
        regression (np.ndarray([..., K, C])): Coordinates of keypoints, where K
            is the joint number and C is the dimension. Example shapes are:
            - [N, K, C]: a batch of keypoints where N is the batch size.
            - [N, T, K, C]: a batch of pose sequences, where T is the frame
                number.
        flip_pairs (list[tuple()]): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).
        center_mode (str): The mode to set the center location on the x-axis
            to flip around. Options are:
            - static: use a static x value (see center_x also)
            - root: use a root joint (see center_index also)
        center_x (float): Set the x-axis location of the flip center. Only used
            when center_mode=static.
        center_index (int): Set the index of the root joint, whose x location
            will be used as the flip center. Only used when center_mode=root.

    Returns:
        tuple: Flipped human joints.

        - regression_flipped (np.ndarray([..., K, C])): Flipped joints.
    """
    assert regression.ndim >= 2, f'Invalid pose shape {regression.shape}'

    allowed_center_mode = {'static', 'root'}
    assert center_mode in allowed_center_mode, 'Get invalid center_mode ' \
        f'{center_mode}, allowed choices are {allowed_center_mode}'

    if center_mode == 'static':
        x_c = center_x
    elif center_mode == 'root':
        assert regression.shape[-2] > center_index
        x_c = regression[..., center_index:center_index + 1, 0]

    regression_flipped = regression.copy()
    # Swap left-right parts
    for left, right in flip_pairs:
        regression_flipped[..., left, :] = regression[..., right, :]
        regression_flipped[..., right, :] = regression[..., left, :]

    # Flip horizontally
    regression_flipped[..., 0] = x_c * 2 - regression_flipped[..., 0]
    return regression_flipped


def process_result(regression_preds, c, s, img_size, score, bbox_ids, image_paths):
    preds, maxvals = get_final_preds(regression_preds, c, s, img_size)
    
    
    all_preds = np.zeros((1, preds.shape[1], 3), dtype=np.float32)
    all_boxes = np.zeros((1, 6), dtype=np.float32)
    all_preds[:, :, 0:2] = preds[:, :, 0:2]
    all_preds[:, :, 2:3] = maxvals
    all_boxes[:, 0:2] = c[0:2]
    all_boxes[:, 2:4] = s[0:2]
    all_boxes[:, 4] = np.prod(s * 200.0)
    all_boxes[:, 5] = score

    result = {}

    result['preds'] = all_preds
    result['boxes'] = all_boxes
    result['image_paths'] = image_paths
    result['bbox_ids'] = bbox_ids
    return result
