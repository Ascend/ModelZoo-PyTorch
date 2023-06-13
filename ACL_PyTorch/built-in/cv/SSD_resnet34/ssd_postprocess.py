# Copyright 2023 Huawei Technologies Co., Ltd
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
from pathlib import Path
from argparse import ArgumentParser
import torch
import numpy as np
from utils import DefaultBoxes, COCODetection
from utils import SSDTransformer
from ssd_r34 import SSD_R34
from pycocotools.cocoeval import COCOeval
from coco import COCO



def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='../coco',
                        help='path to test and training data files')                   
    parser.add_argument('--threshold', '-t', type=float, default='0.19588',
                        help='stop training early at threshold')
    parser.add_argument('--image-size', default=[1200,1200], type=int, nargs='+',
                        help='input image sizes (e.g 1400 1400,1200 1200')  
    parser.add_argument('--strides', default=[3,3,2,2,2,2], type=int, nargs='+',
                        help='stides for ssd model must include 6 numbers')                                                               
    parser.add_argument('--result_dir',  type=str, default=None,
                        help='infer result files')
    return parser.parse_args()



def dboxes_R34_coco(figsize, strides):
	ssd_r34=SSD_R34(81, strides=strides)
	synt_img=torch.rand([1,3]+figsize)
	_,_,feat_size =ssd_r34(synt_img, extract_shapes = True)
	print('Features size: ', feat_size)
	steps=[(int(figsize[0]/fs[0]),int(figsize[1]/fs[1])) for fs in feat_size]
	scales = [(int(s*figsize[0]/300),int(s*figsize[1]/300)) for s in [21, 45, 99, 153, 207, 261, 315]] 
	aspect_ratios =  [[2], [2, 3], [2, 3], [2, 3], [2], [2]] 
	dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
	return dboxes



def coco_eval(cocoGt, inv_map, threshold, result_dir):

	result_infos = {}

	for path in Path(result_dir).iterdir():
		image_id, htot, wtot, output_id = map(int, path.stem.split('_'))
		if image_id not in result_infos:
			result_infos[image_id] = dict(
				res_paths=[None, None, None],
				htot=htot,
				wtot=wtot
			)
		result_infos[image_id]['res_paths'][output_id] = str(path)


	ret = []
	for image_id, infos in result_infos.items():
		loc_path, label_path, prob_path = infos['res_paths']
		htot = infos['htot']
		wtot = infos['wtot']

		loc = np.load(loc_path)
		label = np.load(label_path)
		prob = np.load(prob_path)

		loc = loc.reshape(1, -1, 4)[:, :200, :]
		label = label.reshape(1, -1)[:, :200]
		prob = prob.reshape(1, -1)[:, :200]

		lg = label[label > 0].size 

		loc = loc[:, :lg, :]
		label = label[:, :lg]
		prob = prob[:, :lg]

		for loc_, label_, prob_ in zip(loc[0], label[0], prob[0]):
			ret.append([image_id, loc_[0]*wtot,
                                  loc_[1]*htot,
                                  (loc_[2] - loc_[0])*wtot,
                                  (loc_[3] - loc_[1])*htot,
                                  prob_,
                                  inv_map[label_]])


	cocoDt = cocoGt.loadRes(np.array(ret))

	E = COCOeval(cocoGt, cocoDt, iouType='bbox')
	E.evaluate()
	E.accumulate()
	E.summarize()
	print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))

	return (E.stats[0] >= threshold) 


def main():
    args = parse_args()

    dboxes = dboxes_R34_coco(args.image_size,args.strides)
    val_trans = SSDTransformer(dboxes, (args.image_size[0], args.image_size[1]), val=True)

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")

    cocoGt = COCO(annotation_file=val_annotate)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    inv_map = {v:k for k,v in val_coco.label_map.items()}

    coco_eval(cocoGt, inv_map, args.threshold, args.result_dir)

if __name__ == "__main__":
    main()
