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

from torch.utils.model_zoo import load_url

from ..core import FaceDetector
from ...utils import load_file_from_url

from .net_blazeface import BlazeFace
from .detect import *

models_urls = {
    'blazeface_weights': 'https://github.com/hollance/BlazeFace-PyTorch/blob/master/blazeface.pth?raw=true',
    'blazeface_anchors': 'https://github.com/hollance/BlazeFace-PyTorch/blob/master/anchors.npy?raw=true'
}


class BlazeFaceDetector(FaceDetector):
    def __init__(self, device, path_to_detector=None, path_to_anchor=None, verbose=False,
                 min_score_thresh=0.5, min_suppression_threshold=0.3):
        super(BlazeFaceDetector, self).__init__(device, verbose)

        # Initialise the face detector
        if path_to_detector is None:
            model_weights = load_url(models_urls['blazeface_weights'])
            model_anchors = np.load(load_file_from_url(models_urls['blazeface_anchors']))
        else:
            model_weights = torch.load(path_to_detector)
            model_anchors = np.load(path_to_anchor)

        self.face_detector = BlazeFace()
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.load_anchors_from_npy(model_anchors, device)

        # Optionally change the thresholds:
        self.face_detector.min_score_thresh = min_score_thresh
        self.face_detector.min_suppression_threshold = min_suppression_threshold

        self.face_detector.to(device)
        self.face_detector.eval()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = detect(self.face_detector, image, device=self.device)[0]

        return bboxlist

    def detect_from_batch(self, tensor):
        bboxlists = batch_detect(self.face_detector, tensor, device=self.device)
        return bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
