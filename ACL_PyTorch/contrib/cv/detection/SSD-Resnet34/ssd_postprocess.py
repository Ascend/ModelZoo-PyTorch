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
import os
import argparse
from math import sqrt 
import itertools
import torch
import tqdm
import numpy 
import numpy as np 
from torchvision.ops import batched_nms as NMSOp
from pycocotools.coco import COCO
from PIL import Image
import torch.nn.functional as F
from async_evaluator import AsyncEvaluator
from eval import evaluate_coco



def scale_back_batch(bboxes_in, scores_in):
    """
        Do scale and transform from xywh to ltrb
        suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
    """
    
    bboxes_in = bboxes_in.permute(0, 2, 1)
    scores_in = scores_in.permute(0, 2, 1)

    fig_size = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    fk = fig_size/np.array(steps)
    
    default_boxes = []

    for idx, sfeat in enumerate(feat_size):
        sk1 = scales[idx]/fig_size
        sk2 = scales[idx+1]/fig_size
        sk3 = sqrt(sk1*sk2)
        all_sizes = [(sk1, sk1), (sk3, sk3)]
        for alpha in aspect_ratios[idx]:
            w, h = sk1*sqrt(alpha), sk1/sqrt(alpha)
            all_sizes.append((w, h))
            all_sizes.append((h, w))
        for w, h in all_sizes:
            for i, j in itertools.product(range(sfeat), repeat=2):
                cx, cy = (j+0.5)/fk[idx], (i+0.5)/fk[idx]
                default_boxes.append((cx, cy, w, h))

    dboxes = torch.tensor(default_boxes, dtype=torch.float).cpu()
    dboxes.clamp_(min=0, max=1)
    bboxes_in[:, :, :2] = 0.1*bboxes_in[:, :, :2]
    bboxes_in[:, :, 2:] = 0.2*bboxes_in[:, :, 2:]
    
    bboxes_in[:, :, :2] = bboxes_in[:, :, :2]*dboxes.unsqueeze(dim=0).cpu()[:, :, 2:]\
                          + dboxes.unsqueeze(dim=0).cpu()[:, :, :2]
    bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp()*dboxes.unsqueeze(dim=0).cpu()[:, :, 2:]
    
    l, t, r, b = bboxes_in[:, :, 0] - 0.5*bboxes_in[:, :, 2],\
                 bboxes_in[:, :, 1] - 0.5*bboxes_in[:, :, 3],\
                 bboxes_in[:, :, 0] + 0.5*bboxes_in[:, :, 2],\
                 bboxes_in[:, :, 1] + 0.5*bboxes_in[:, :, 3]
    
    bboxes_in[:, :, 0] = l
    bboxes_in[:, :, 1] = t
    bboxes_in[:, :, 2] = r
    bboxes_in[:, :, 3] = b
    
    return bboxes_in, F.softmax(scores_in, dim=-1)


def main():
    ret = []
    evaluator = AsyncEvaluator(num_threads=1)
    parser = argparse.ArgumentParser(description='postprocess of SSD pytorch model')
    parser.add_argument("--val_annotation", default="bbox_only_instances_val2017.json")
    parser.add_argument("--bin_path", default="result")
    args = parser.parse_args()
    val_annotation = args.val_annotation
    bin_path = args.bin_path
    cocoGT = COCO(annotation_file=val_annotation)
    data = cocoGT.dataset
    images = {}
    label_map = {}
    inv_map = {}
    cnt = 0
    for cat in data['categories']:
        cnt += 1
        label_map[cat['id']] = cnt
        inv_map[cnt] = cat['id']

        
    for img in data["images"]:
        img_id = img["id"]
        img_name = img["file_name"]
        img_size = (img["height"], img["width"])
        images[img_id] = (img_name, img_size, [])

    for bboxes in data["annotations"]:
        img_id = bboxes["image_id"]
        category_id = bboxes["category_id"]
        bbox = bboxes["bbox"]
        bbox_label = label_map[bboxes["category_id"]]
        images[img_id][2].append((bbox, bbox_label))

    for k, v in tqdm.tqdm(list(images.items())):
        if len(v[2]) == 0:
            images.pop(k)
        else:
            file_name = os.path.join(bin_path, v[0].split('.')[0])
            bbox_file = file_name + '_0' + '.bin'
            score_file = file_name + '_1' + '.bin'
            bboxes = np.fromfile(bbox_file, dtype="float32").reshape(1, 4, 8732)
            scores = np.fromfile(score_file, dtype="float32").reshape(1, 81, 8732)
            bboxes = torch.from_numpy(bboxes)
            scores = torch.from_numpy(scores)
            bboxes, scores = scale_back_batch(bboxes, scores)
            bboxes = (bboxes + 1)/2
            bboxes = bboxes.view(-1, 4)
            scores = scores.view(-1, 81)[..., 1:]
            max_scores, _ = scores.max(-1)
            keep_inds = (max_scores > 0.05).nonzero(as_tuple=False).view(-1)
            bboxes = bboxes[keep_inds, :]
            scores = scores[keep_inds, :]
            if bboxes.shape[0] > 200:
                max_scores, _ = scores.max(-1)
                _, topk_inds = max_scores.topk(200)
                bboxes = bboxes[topk_inds, :]
                scores = scores[topk_inds, :]
            class_idxs = torch.arange(80, dtype=torch.long)[None, :].repeat(bboxes.shape[0], 1).view(-1)
            bboxes = bboxes[:, None, :].repeat(1, 80, 1)
            bboxes = bboxes.view(-1, 4)
            scores = scores.view(-1)
            keep_inds = NMSOp(bboxes, scores, class_idxs, 0.5)
            bboxes = bboxes[keep_inds]
            bboxes = bboxes*2 - 1
            scores = scores[keep_inds]
            class_idxs = class_idxs[keep_inds]
            if bboxes.shape[0] > 200:
                _, topk_inds = scores.topk(200)
                bboxes = bboxes[topk_inds, :]
                scores = scores[topk_inds]
                class_idxs = class_idxs[topk_inds]
            bboxes = bboxes.numpy()
            scores = scores.numpy()
            class_idxs = class_idxs.numpy()
            htot, wtot = v[1][0], v[1][1]
            for loc_, label_, prob_ in zip(bboxes, class_idxs, scores):
                ret.append([k, loc_[0]*wtot,
                            loc_[1]*htot,
                            (loc_[2] - loc_[0])*wtot,
                            (loc_[3] - loc_[1])*htot,
                            prob_,
                            inv_map[(label_+1)]])
    ret = np.array(ret).astype(np.float32)
    final_results = ret
    evaluator.submit_task(0, evaluate_coco, final_results, cocoGT, 0, 0.25)


if __name__ == "__main__":
    main()