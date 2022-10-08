
# Copyright 2020 Huawei Technologies Co., Ltd
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

import cv2
import numpy as np


class draw:

    """Drawing utils for images"""

    @staticmethod
    def bbox(img, bb, color=(0, 255, 0)):
        """draw a bounding box on image
        @img: OpenCV image
        @bb: bounding box, assuming all the boundary conditions are
        satisfied
        @color: color of the bounding box
        """

        img_out = np.copy(img)
        img_out = np.ascontiguousarray(img_out)
        x1, y1 = int(bb.x1), int(bb.y1)
        x2, y2 = int(bb.x2), int(bb.y2)

        img_out = cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

        return img_out
