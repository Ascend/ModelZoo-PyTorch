# Copyright 2022 Huawei Technologies Co., Ltd
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

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append('./TransPose')
from lib.config import cfg
from lib.config import update_config
from lib.core.inference import get_final_preds
from lib.dataset.coco import COCODataset
from lib.utils.transforms import flip_back
from lib.utils.utils import create_logger

parser = argparse.ArgumentParser(description='Test keypoints network')
# general
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    type=str,
                    default='TransPose/experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc3_mh8.yaml'
                    )

parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
parser.add_argument('--dump_dir', dest='dump_dir',
                    help='dump dir for bin files',
                    required=True,
                    type=str)
parser.add_argument('--dump_dir_flip', dest='dump_dir_flip',
                    help='dump dir for flip bin files',
                    required=True,
                    type=str)

opt = parser.parse_args()


def get_output_data(dump_dir, idx, dtype=np.float32):
    output_shape = [1, 17, 64, 48]
    input_file = os.path.join(dump_dir, "{:0>12d}_0.bin".format(idx))
    input_data = np.fromfile(input_file, dtype=dtype).reshape(output_shape)

    input_data = torch.tensor(input_data, dtype=torch.float32)

    return input_data


def postprocess(config):
    logger, final_output_dir, tb_log_dir = create_logger(
        config, opt.cfg, 'valid')

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = COCODataset(
        config, config.DATASET.ROOT, config.DATASET.TEST_SET, False
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    num_samples = len(valid_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    for idx, (image, _, _, meta) in tqdm(enumerate(valid_loader)):
        outputs = get_output_data(opt.dump_dir, idx)
        outputs_flipped = get_output_data(opt.dump_dir_flip, idx)
        output_flipped = flip_back(outputs_flipped.cpu().numpy(),
                                   valid_dataset.flip_pairs)
        output_flipped = torch.from_numpy(output_flipped.copy())
        output = (outputs + output_flipped) * 0.5

        c = meta['center'].numpy()
        s = meta['scale'].numpy()
        score = meta['score'].numpy()
        num_images = image.size(0)

        preds, maxvals = get_final_preds(
            config, output.clone().cpu().numpy(), c, s)

        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        # double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
        all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
        all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
        all_boxes[idx:idx + num_images, 5] = score
        image_path.extend(meta['image'])

        idx += num_images
        prefix = '{}_{}'.format(os.path.join(final_output_dir, 'result_valid'), idx)

    name_values, perf_indicator = valid_dataset.evaluate(
        config, all_preds, final_output_dir, all_boxes, image_path,
        filenames, imgnums
    )

    model_name = config.MODEL.NAME
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(logger, name_value, model_name)
    else:
        _print_name_value(logger, name_values, model_name)


def _print_name_value(log, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    log.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    log.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    log.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


coco_part_labels = [
    'nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
    'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
    'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r'
]
coco_part_idx = {
    b: a for a, b in enumerate(coco_part_labels)
}
coco_part_orders = [
    ('nose', 'eye_l'), ('eye_l', 'eye_r'), ('eye_r', 'nose'),
    ('eye_l', 'ear_l'), ('eye_r', 'ear_r'), ('ear_l', 'sho_l'),
    ('ear_r', 'sho_r'), ('sho_l', 'sho_r'), ('sho_l', 'hip_l'),
    ('sho_r', 'hip_r'), ('hip_l', 'hip_r'), ('sho_l', 'elb_l'),
    ('elb_l', 'wri_l'), ('sho_r', 'elb_r'), ('elb_r', 'wri_r'),
    ('hip_l', 'kne_l'), ('kne_l', 'ank_l'), ('hip_r', 'kne_r'),
    ('kne_r', 'ank_r')
]
VIS_CONFIG = {
    'COCO': {
        'part_labels': coco_part_labels,
        'part_idx': coco_part_idx,
        'part_orders': coco_part_orders
    }
}


def add_joints(image, joints, color, dataset='COCO'):
    part_idx = VIS_CONFIG[dataset]['part_idx']
    part_orders = VIS_CONFIG[dataset]['part_orders']

    def link(a, b, color):
        if part_idx[a] < joints.shape[0] and part_idx[b] < joints.shape[0]:
            jointa = joints[part_idx[a]]
            jointb = joints[part_idx[b]]
            if jointa[2] > 0 and jointb[2] > 0:
                cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                    color,
                    2
                )

    # add joints
    for joint in joints:
        if joint[2] > 0:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 1, color, 2)

    # add link
    for pair in part_orders:
        link(pair[0], pair[1], color)

    return image


if __name__ == '__main__':
    update_config(cfg, opt)
    postprocess(cfg)
