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

import os
import os.path as osp
import numpy as np
from pycocotools.coco import COCO
from utils.pose_utils import process_bbox
from config import cfg

class MPII:

    def __init__(self, data_split):
        self.data_split = data_split
        self.img_dir = osp.join('..', 'data', 'MPII')
        self.train_annot_path = osp.join('..', 'data', 'MPII', 'annotations', 'train.json')
        self.joint_num = 16
        self.joints_name = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Thorax', 'Neck', 'Head', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist')
        self.joints_have_depth = False
        self.root_idx = self.joints_name.index('Pelvis')
        self.data = self.load_data()

    def load_data(self):
        
        if self.data_split == 'train':
            db = COCO(self.train_annot_path)
        else:
            print('Unknown data subset')
            assert 0

        data = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            img = db.loadImgs(ann['image_id'])[0]
            width, height = img['width'], img['height']

            if ann['num_keypoints'] == 0:
                continue
            
            bbox = process_bbox(ann['bbox'], width, height)
            if bbox is None: continue
            area = bbox[2]*bbox[3]

            # joints and vis
            joint_img = np.array(ann['keypoints']).reshape(self.joint_num,3)
            joint_vis = joint_img[:,2].copy().reshape(-1,1)
            joint_img[:,2] = 0
            root_img = joint_img[self.root_idx]
            root_vis = joint_vis[self.root_idx]

            imgname = db.imgs[ann['image_id']]['file_name']
            img_path = osp.join(self.img_dir, imgname)
            data.append({
                'img_path': img_path,
                'bbox': bbox,
                'area': area,
                'root_img': root_img, # [org_img_x, org_img_y, 0]
                'root_vis': root_vis,
                'f': np.array([1500, 1500]) # dummy value
            })

        return data
