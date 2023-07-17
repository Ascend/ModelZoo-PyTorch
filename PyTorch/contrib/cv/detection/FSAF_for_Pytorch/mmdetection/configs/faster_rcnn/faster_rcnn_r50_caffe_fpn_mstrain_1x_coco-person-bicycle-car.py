# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#
import os

_base_ = './faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
model = dict(roi_head=dict(bbox_head=dict(num_classes=3)))
classes = ('person', 'bicycle', 'car')
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

current_path = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_path, '../../../url.ini'), 'r') as _f:
    _content = _f.read()
    faster_rcnn_r50_caffe_url = _content.split('faster_rcnn_r50_caffe_url=')[1].split('\n')[0]
load_from = faster_rcnn_r50_caffe_url  # noqa
