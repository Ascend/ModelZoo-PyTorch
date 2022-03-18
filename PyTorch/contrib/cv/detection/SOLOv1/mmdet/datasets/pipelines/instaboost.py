# Copyright 2021 Huawei Technologies Co., Ltd
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

import numpy as np

from ..registry import PIPELINES


@PIPELINES.register_module
class InstaBoost(object):
    """
    Data augmentation method in paper "InstaBoost: Boosting Instance
    Segmentation Via Probability Map Guided Copy-Pasting"
    Implementation details can refer to https://github.com/GothicAi/Instaboost.
    """

    def __init__(self,
                 action_candidate=('normal', 'horizontal', 'skip'),
                 action_prob=(1, 0, 0),
                 scale=(0.8, 1.2),
                 dx=15,
                 dy=15,
                 theta=(-1, 1),
                 color_prob=0.5,
                 hflag=False,
                 aug_ratio=0.5):
        try:
            import instaboostfast as instaboost
        except ImportError:
            raise ImportError(
                'Please run "pip install instaboostfast" '
                'to install instaboostfast first for instaboost augmentation.')
        self.cfg = instaboost.InstaBoostConfig(action_candidate, action_prob,
                                               scale, dx, dy, theta,
                                               color_prob, hflag)
        self.aug_ratio = aug_ratio

    def _load_anns(self, results):
        labels = results['ann_info']['labels']
        masks = results['ann_info']['masks']
        bboxes = results['ann_info']['bboxes']
        n = len(labels)

        anns = []
        for i in range(n):
            label = labels[i]
            bbox = bboxes[i]
            mask = masks[i]
            x1, y1, x2, y2 = bbox
            bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
            anns.append({
                'category_id': label,
                'segmentation': mask,
                'bbox': bbox
            })

        return anns

    def _parse_anns(self, results, anns, img):
        gt_bboxes = []
        gt_labels = []
        gt_masks_ann = []
        for ann in anns:
            x1, y1, w, h = ann['bbox']
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            gt_bboxes.append(bbox)
            gt_labels.append(ann['category_id'])
            gt_masks_ann.append(ann['segmentation'])
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
        results['ann_info']['labels'] = gt_labels
        results['ann_info']['bboxes'] = gt_bboxes
        results['ann_info']['masks'] = gt_masks_ann
        results['img'] = img
        return results

    def __call__(self, results):
        img = results['img']
        anns = self._load_anns(results)
        if np.random.choice([0, 1], p=[1 - self.aug_ratio, self.aug_ratio]):
            try:
                import instaboostfast as instaboost
            except ImportError:
                raise ImportError('Please run "pip install instaboostfast" '
                                  'to install instaboostfast first.')
            anns, img = instaboost.get_new_data(
                anns, img, self.cfg, background=None)
        results = self._parse_anns(results, anns, img)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(cfg={}, aug_ratio={})').format(self.cfg, self.aug_ratio)
        return repr_str
