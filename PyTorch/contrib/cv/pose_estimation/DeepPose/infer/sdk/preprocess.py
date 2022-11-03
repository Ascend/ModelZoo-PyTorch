# Copyright 2022 Huawei Technologies Co., Ltd
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
import json_tricks as json
import os
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval


evaluation = {'interval':10, 'metric':'mAP', 'key_indicator':'AP'}

channel_cfg = {
    'num_output_channels':17,
    'dataset_joints' : 17,
    'dataset_channel':[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    'inference_channel':[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ]}

data_cfg = {
    'image_size':np.array([192, 256]),
    'heatmap_size':np.array([48, 64]),
    'num_output_channels':channel_cfg['num_output_channels'],
    'num_joints':channel_cfg['dataset_joints'],
    'dataset_channel':channel_cfg['dataset_channel'],
    'inference_channel':channel_cfg['inference_channel'],
    'nms_thr':1.0,
    'oks_thr':0.9,
    'vis_thr':0.2,
    'det_bbox_thr':0.0,
    'bbox_file':'person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
    'flip_pairs':[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
}

img_prefix='val2017'

sigmas = np.array([
        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
        .87, .87, .89, .89
    ]) / 10.0

def get_mapping_id_name(imgs):
    """
    Args:
        imgs (dict): dict of image info.

    Returns:
        tuple: Image name & id mapping dicts.

        - id2name (dict): Mapping image id to name.
        - name2id (dict): Mapping image name to id.
    """
    id2name = {}
    name2id = {}
    for image_id, image in imgs.items():
        file_name = image['file_name']
        id2name[image_id] = file_name
        name2id[file_name] = image_id

    return id2name, name2id

def _xywh2cs(x, y, w, h):
    """This encodes bbox(x,y,w,w) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - center (np.ndarray[float32](2,)): center of the bbox (x, y).
        - scale (np.ndarray[float32](2,)): scale of the bbox w & h.
    """
    aspect_ratio = data_cfg['image_size'][0] / data_cfg[
        'image_size'][1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
    # padding to include proper amount of context
    scale = scale * 1.25
    return center, scale

def load_coco_person_detection_results(data_root, id2name):
    """Load coco person detection results."""
    dataset_name = 'coco'
    num_joints = data_cfg['num_joints']
    all_boxes = None
    bbox_file =  os.path.join(data_root, data_cfg['bbox_file'])
    with open(bbox_file, 'r') as f:
        all_boxes = json.load(f)

    if not all_boxes:
        raise ValueError('=> Load %s fail!' % bbox_file)

    print('=> Total boxes: {len(all_boxes)}')
    print(data_root)
    kpt_db = []
    bbox_id = 0
    for det_res in all_boxes:
        if det_res['category_id'] != 1:
            continue
        image_file = os.path.join(data_root, img_prefix)
        image_file = os.path.join(image_file, id2name[det_res['image_id']])
        print(image_file)
        box = det_res['bbox']
        score = det_res['score']
        if score < data_cfg['det_bbox_thr']:
            continue
        center, scale = _xywh2cs(*box[:4])
        kpt_db.append({
            'image_file': image_file,
            'center': center,
            'scale': scale,
            'rotation': 0,
            'bbox': box[:4],
            'bbox_score': score,
            'dataset': dataset_name,
            'bbox_id': bbox_id
        })
        bbox_id = bbox_id + 1
    print('=> Total boxes after filter '
            'low score@{det_bbox_thr}: {bbox_id}')
    return kpt_db

def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]
    return rotated_pt


def affine_transform(pt, trans_mat):
    """Apply an affine transformation to the points.

    Args:
        pt (np.ndarray): a 2 dimensional point to be transformed
        trans_mat (np.ndarray): 2x3 matrix of an affine transform

    Returns:
        np.ndarray: Transformed points. 
    """
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(trans_mat, new_pt)
    return new_pt[:2]
    
    
def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point. 
    """
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)
    return third_pt


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False,
                         scale_ratio=200.0):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ]): Size of the destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)
        scale_ratio(float): pixel std of MPII is 200

    Returns:
        np.ndarray: The transform matrix.
    """ 
    scale_tmp = scale * scale_ratio
    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans    
    

def get_img(args, coco, id2name):
    img_ids = coco.getImgIds()
    num_images = len(img_ids)
    db = load_coco_person_detection_results(args.data_root, id2name)
    print(f'=> num_images: {num_images}')
    print(f'=> load {len(db)} samples')   
    

    for idx in range(len(db)):
        print("processing: {}",idx/len(db))    
        kpt = db[idx]
        img_name = kpt['image_file']
        c = kpt['center']
        s = kpt['scale']
        r = kpt['rotation'] # rotation
        score = kpt['bbox_score']
        bbox_id = kpt['bbox_id']
        img_bgr = cv2.imread(img_name, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # transform image format from BGR to RGB
        image_size = (192, 256) # input shape of model
        trans = get_affine_transform(c, s, r, image_size) # get affine transform matrix
        img_trans = cv2.warpAffine(
                    img_rgb,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags = cv2.INTER_LINEAR)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_mean_std = (img_trans.astype(np.float32) / 255 - mean) / std
        img_eval = img_mean_std.reshape((1,) + img_mean_std.shape)
        img_eval = img_eval.transpose(0, 3, 1, 2) # turn image format into NCHW
        yield img_eval, c, s, img_name, score,bbox_id 

    
