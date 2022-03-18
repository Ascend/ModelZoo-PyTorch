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
# ============================================================================
""" This script transforms and saves bbox coordinates into a pickle object for easy loading. """


import os.path as osp
import json, pickle
import sys

import numpy as np

COCO_ROOT = osp.join('.', 'data/coco/')

annotation_file = 'instances_train2017.json'
annotation_path = osp.join(COCO_ROOT, 'annotations/', annotation_file)

dump_file = 'weights/bboxes.pkl'

with open(annotation_path, 'r') as f:
	annotations_json = json.load(f)

annotations = annotations_json['annotations']
images = annotations_json['images']
images = {image['id']: image for image in images}
bboxes = []

for ann in annotations:
	image = images[ann['image_id']]
	w,h = (image['width'], image['height'])
	
	if 'bbox' in ann:
		bboxes.append([w, h] + ann['bbox'])

with open(dump_file, 'wb') as f:
	pickle.dump(bboxes, f)
