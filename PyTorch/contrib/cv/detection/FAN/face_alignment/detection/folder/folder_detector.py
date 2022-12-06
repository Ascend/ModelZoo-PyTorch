# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import torch

from ..core import FaceDetector


class FolderDetector(FaceDetector):
    '''This is a simple helper module that assumes the faces were detected already
        (either previously or are provided as ground truth).

        The class expects to find the bounding boxes in the same format used by
        the rest of face detectors, mainly ``list[(x1,y1,x2,y2),...]``.
        For each image the detector will search for a file with the same name and with one of the
        following extensions: .npy, .t7 or .pth

    '''

    def __init__(self, device, path_to_detector=None, verbose=False):
        super(FolderDetector, self).__init__(device, verbose)

    def detect_from_image(self, tensor_or_path):
        # Only strings supported
        if not isinstance(tensor_or_path, str):
            raise ValueError

        base_name = os.path.splitext(tensor_or_path)[0]

        if os.path.isfile(base_name + '.npy'):
            detected_faces = np.load(base_name + '.npy')
        elif os.path.isfile(base_name + '.t7'):
            detected_faces = torch.load(base_name + '.t7')
        elif os.path.isfile(base_name + '.pth'):
            detected_faces = torch.load(base_name + '.pth')
        else:
            raise FileNotFoundError

        if not isinstance(detected_faces, list):
            raise TypeError

        return detected_faces

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
