# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

logger = logging.getLogger(__name__)


class JointsLambdaDataset(Dataset):
    def __init__(self, cfg, image_dir, annotation_file, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale


    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec_A = copy.deepcopy(self.db[idx])

        if db_rec_A['min_dist_db_rec'] is not None:
            db_rec_B = copy.deepcopy(db_rec_A['min_dist_db_rec'])
        else:
            db_rec_B = copy.deepcopy(db_rec_A)

        input_A, joints_A, joints_vis_A, meta_A, joints_B, joints_vis_B, meta_B = self.getitem_helper(db_rec_A, db_rec_B)

        return self.getitem_prepare(input_A, joints_A, joints_vis_A, meta_A, joints_B, joints_vis_B, meta_B)
                    
    # ------------------------------------------------------------
    def getitem_prepare(self, input, joints_foreground, joints_vis_foreground, meta_foreground, joints_background, joints_vis_background, meta_background):
        if self.transform:
            input = self.transform(input)

        target_foreground, target_weight_foreground = self.generate_target(joints_foreground, joints_vis_foreground)
        target_foreground = torch.from_numpy(target_foreground)
        target_weight_foreground = torch.from_numpy(target_weight_foreground)
        meta_foreground['joints'] = joints_foreground
        meta_foreground['joints_vis'] = joints_vis_foreground

        target_background, target_weight_background = self.generate_target(joints_background, joints_vis_background)
        target_background = torch.from_numpy(target_background)
        target_weight_background = torch.from_numpy(target_weight_background)
        meta_background['joints'] = joints_background
        meta_background['joints_vis'] = joints_vis_background

        return input, target_foreground, target_weight_foreground, meta_foreground, target_background, target_weight_background, meta_background
 
    def getitem_helper(self, db_rec, another_db_rec=None):
        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        another_joints = another_db_rec['joints_3d']
        another_joints_vis = another_db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                another_joints, another_joints_vis = fliplr_joints(
                    another_joints, another_joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        joints, joints_vis = self.get_transformed_joints(np.copy(joints), np.copy(joints_vis), trans)
        another_joints, another_joints_vis = self.get_transformed_joints(np.copy(another_joints), np.copy(another_joints_vis), trans)

        # -------------------------------
        if another_joints_vis[:, 0].sum() <= 3:
            another_joints = np.copy(joints)
            another_joints_vis = np.copy(joints_vis)
            another_db_rec = db_rec

        # -------------------------------
        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'annotation_id': db_rec['annotation_id']
        }

        image_file = another_db_rec['image']
        filename = another_db_rec['filename'] if 'filename' in another_db_rec else ''
        imgnum = another_db_rec['imgnum'] if 'imgnum' in another_db_rec else ''

        another_meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': another_joints,
            'joints_vis': another_joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'annotation_id': another_db_rec['annotation_id']
        }

        return input, joints, joints_vis, meta, another_joints, another_joints_vis, another_meta

    def get_transformed_joints(self, joints, joints_vis, trans):
        for i in range(self.num_joints):
            joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

            if joints[i, 0] > self.image_size[0] or joints[i, 1] > self.image_size[1] or joints[i, 0] < 0 or joints[i, 1] < 0:
                joints[i, 0] = 0.0; joints[i, 1] = 0.0 
                joints_vis[i, 0] = 0; joints_vis[i, 1] = 0 
        
        return joints, joints_vis

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def compute_iou_seg(self, seg1, seg2):
        intersection = np.logical_and(seg1, seg2)
        union = np.logical_or(seg1, seg2)
        iou = np.sum(intersection)/np.sum(union)
        return iou

    ## bbox format = x, y, w, h
    def compute_iou(self, bbox_1, bbox_2):

        x1_l = bbox_1[0]
        x1_r = bbox_1[0] + bbox_1[2]
        y1_t = bbox_1[1]
        y1_b = bbox_1[1] + bbox_1[3]
        w1   = bbox_1[2]
        h1   = bbox_1[3]

        x2_l = bbox_2[0]
        x2_r = bbox_2[0] + bbox_2[2]
        y2_t = bbox_2[1]
        y2_b = bbox_2[1] + bbox_2[3]
        w2   = bbox_2[2]
        h2   = bbox_2[3]

        xi_l = max(x1_l, x2_l)
        xi_r = min(x1_r, x2_r)
        yi_t = max(y1_t, y2_t)
        yi_b = min(y1_b, y2_b)

        width  = max(0, xi_r - xi_l)
        height = max(0, yi_b - yi_t)
        a1 = w1 * h1
        a2 = w2 * h2

        if float(a1 + a2 - (width * height)) == 0:
            return 0
        else:
            iou = (width * height) / float(a1 + a2 - (width * height))

        return iou
