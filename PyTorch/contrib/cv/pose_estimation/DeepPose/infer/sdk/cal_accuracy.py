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
# ============================================================================

import copy
import numpy as np
import os
import json_tricks as json
from collections import defaultdict
from collections import OrderedDict
from preprocess import data_cfg
from xtcocotools.cocoeval import COCOeval


sigmas = np.array([
        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
        .87, .87, .89, .89
    ]) / 10.0

def sort_and_unique_bboxes(kpts, key='bbox_id'):
    """sort kpts and remove the repeated ones."""
    for img_id, persons in kpts.items():
        num = len(persons)
        kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
        for i in range(num - 1, 0, -1):
            if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                del kpts[img_id][i]

    return kpts


def oks_iou(g, d, a_g, a_d, sigmas=None, vis_thr=None):
    """Calculate oks ious.

    Args:
        g: Ground truth keypoints.
        d: Detected keypoints.
        a_g: Area of the ground truth object.
        a_d: Area of the detected object.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.

    Returns:
        list: The oks ious.
    """
    vars = (sigmas * 2)**2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d), dtype=np.float32)
    for n_d in range(0, len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if vis_thr is not None:
            ind = list(vg > vis_thr) and list(vd > vis_thr)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) != 0 else 0.0
    return ious


def oks_nms(kpts_db, thr, sigmas=None, vis_thr=None):
    """OKS NMS implementations.

    Args:
        kpts_db: keypoints.
        thr: Retain overlap < thr.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.

    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    scores = np.array([k['score'] for k in kpts_db])
    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        inds = np.where(oks_ovr <= thr)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep


def coco_keypoint_results_one_category_kernel(data_pack):
    """Get coco keypoint results."""
    cat_id = data_pack['cat_id']
    keypoints = data_pack['keypoints']
    cat_results = []

    for img_kpts in keypoints:
        if len(img_kpts) == 0:
            continue

        _key_points = np.array(
            [img_kpt['keypoints'] for img_kpt in img_kpts])
        key_points = _key_points.reshape(-1,
                                            data_cfg['num_joints'] * 3)

        result = [{
            'image_id': img_kpt['image_id'],
            'category_id': cat_id,
            'keypoints': key_point.tolist(),
            'score': float(img_kpt['score']),
            'center': img_kpt['center'].tolist(),
            'scale': img_kpt['scale'].tolist()
        } for img_kpt, key_point in zip(img_kpts, key_points)]

        cat_results.extend(result)

    return cat_results


def write_coco_keypoint_results(keypoints, res_file, coco):
        """Write results into a json file."""
        cats = [
            cat['name'] for cat in coco.loadCats(coco.getCatIds())
        ]
        classes = ['__background__'] + cats

        class_to_coco_ind = dict(zip(cats, coco.getCatIds()))

        data_pack = [{
            'cat_id': class_to_coco_ind[cls],
            'cls_ind': cls_ind,
            'cls': cls,
            'ann_type': 'keypoints',
            'keypoints': keypoints
        } for cls_ind, cls in enumerate(classes)
                     if not cls == '__background__']

        results = coco_keypoint_results_one_category_kernel(data_pack[0])

        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)


def do_python_keypoint_eval(coco, res_file):
    """Keypoint evaluation using COCOAPI."""
    coco_det = coco.loadRes(res_file)
    coco_eval = COCOeval(coco, coco_det, 'keypoints', sigmas)
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats_names = [
        'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
        'AR .75', 'AR (M)', 'AR (L)'
    ]

    info_str = list(zip(stats_names, coco_eval.stats))

    return info_str


def evaluate(outputs, res_folder, data_cfg, name2id, coco, args):
    """Evaluate coco keypoint results. The pose prediction results will be
    saved in `${res_folder}/result_keypoints.json`.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        outputs (list(dict))
            :preds (np.ndarray[N,K,3]): The first two dimensions are
                coordinates, score is the third dimension of the array.
            :boxes (np.ndarray[N,6]): [center[0], center[1], scale[0]
                , scale[1],area, score]
            :image_paths (list[str]): For example, ['data/coco/val2017
                /000000393226.jpg']
            :heatmap (np.ndarray[N, K, H, W]): model output heatmap
            :bbox_id (list(int)).
        res_folder (str): Path of directory to save the results.
        metric (str | list[str]): Metric to be performed. Defaults: 'mAP'.

    Returns:
        dict: Evaluation results for evaluation metric.
    """

    res_file = os.path.join(res_folder, 'result_keypoints.json')

    kpts = defaultdict(list)

    for output in outputs:
        preds = output['preds']
        boxes = output['boxes']
        image_paths = output['image_paths']
        bbox_ids = output['bbox_ids']

        img_prefix=f'{args.data_root}/val2017/'
        image_id = name2id[image_paths[len(img_prefix):]]
        kpts[image_id].append({
            'keypoints': preds[0],
            'center': boxes[0][0:2],
            'scale': boxes[0][2:4],
            'area': boxes[0][4],
            'score': boxes[0][5],
            'image_id': image_id,
            'bbox_id': bbox_ids
        })
    kpts = sort_and_unique_bboxes(kpts)

    # rescoring and oks nms
    num_joints = data_cfg['num_joints'] #self.ann_info['num_joints']
    vis_thr = data_cfg['vis_thr']#self.vis_thr
    oks_thr = data_cfg['oks_thr'] #self.oks_thr
    valid_kpts = []
    for image_id in kpts.keys():
        img_kpts = kpts[image_id]
        for n_p in img_kpts:
            box_score = n_p['score']
            kpt_score = 0
            valid_num = 0
            for n_jt in range(0, num_joints):
                t_s = n_p['keypoints'][n_jt][2]
                if t_s > vis_thr:
                    kpt_score = kpt_score + t_s
                    valid_num = valid_num + 1
            if valid_num != 0:
                kpt_score = kpt_score / valid_num
            # rescoring
            n_p['score'] = kpt_score * box_score
        keep = oks_nms(list(img_kpts), oks_thr, sigmas = sigmas)
        valid_kpts.append([img_kpts[_keep] for _keep in keep])
    write_coco_keypoint_results(valid_kpts, res_file, coco)
    info_str = do_python_keypoint_eval(coco, res_file)
    name_value = OrderedDict(info_str)
    return name_value
