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
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
from utils.box_utils import decode


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, cfg):
        self.num_classes = num_classes
        self.background_label = bkg_label

        self.variance = cfg['variance']

    def forward(self, predictions, prior):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        """
        if args.device.startswith('cuda'):
            torch.cuda.set_device(args.device)
        elif args.device.startswith('npu'):
            torch.npu.set_device(args.device)
        else:
            pass
        """
        loc, conf = predictions

        loc_data = loc.data
        conf_data = conf.data
        prior_data = prior.data
        num = loc_data.size(0)  # batch size
        self.num_priors = prior_data.size(0)
        self.boxes = torch.zeros(1, self.num_priors, 4)
        self.scores = torch.zeros(1, self.num_priors, self.num_classes)

        self.boxes = self.boxes.npu()
        self.scores = self.scores.npu()

        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.unsqueeze(0)

        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes)
            self.boxes.expand_(num, self.num_priors, 4)
            self.scores.expand_(num, self.num_priors, self.num_classes)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()

            self.boxes[i] = decoded_boxes
            self.scores[i] = conf_scores

        return self.boxes, self.scores
