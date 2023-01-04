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

from __future__ import print_function
import os
import argparse
import torch
import numpy as np
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer


cfg = {
    'name': 'FaceBoxes', 
    'min_sizes': [[32, 64, 128], [256], [512]], 
    'steps': [32, 64, 128], 'variance': [0.1, 0.2], 
    'clip': False, 
    'loc_weight': 2.0, 
    'gpu_train': True
}

def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


parser = argparse.ArgumentParser(description='FaceBoxes')

parser.add_argument('--save_folder', default='FDDB_Evaluation/', type=str, help='Dir to save results')
parser.add_argument('--prep_info', default='prep/')
parser.add_argument('--prep_folder', default='benchmark_tools/result/dumpOutput_device0/')

parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')

parser.add_argument('--dataset', default='FDDB', type=str, choices=['AFW', 'PASCAL', 'FDDB'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--show_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()



if __name__ == '__main__':
    
    _t = {'forward_pass': Timer(), 'misc': Timer()}
    print('1')
    # save file
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    fw = open(os.path.join(args.save_folder, 'FDDB_dets.txt'), 'w')
    print('2')
    num_images=0
    # prep_info
    prepinfo_list = os.path.join(args.prep_info, 'FDDB.txt')
    with open(prepinfo_list, 'r') as fr:
        
    
    # testing begin
        for prep_info in fr:
            
            _t['misc'].tic()
            num_images= num_images+1
            print(prep_info)
            #input info
            img_name,im_height,im_width,resize=prep_info.split(' ')
            
            im_height = np.float32(im_height)
            im_width = np.float32(im_width)
            resize = np.float32(resize)
            
            scale = torch.Tensor([im_width, im_height, im_width, im_height])
            
            #input loc conf
            img_bin_1 = os.path.join(args.prep_folder,img_name+'_0.bin')
            img_bin_2 = os.path.join(args.prep_folder,img_name+'_1.bin')
            buf_1 = np.fromfile(img_bin_1, dtype="float32")
            buf_2 = np.fromfile(img_bin_2, dtype="float32") 
            conf = np.reshape(buf_2, [1, 21824, 2])
            loc = np.reshape(buf_1, [1, 21824, 4])
            
            loc = torch.Tensor(loc)
            conf = torch.Tensor(conf)

            priorbox = PriorBox(cfg, image_size=(1024, 1024))
            priors = priorbox.forward()
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, [0.1, 0.2])
            
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    
            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            scores = scores[inds]
    
            # keep top-K before NMS
            order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            scores = scores[order]
    
            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            #keep = py_cpu_nms(dets, args.nms_threshold)
            keep = nms(dets, args.nms_threshold,force_cpu=True)
            dets = dets[keep, :]
    
            # keep top-K faster NMS
            dets = dets[:args.keep_top_k, :]
            
            _t['misc'].toc()
            # save dets
            if args.dataset == "FDDB":
                fw.write('{:s}\n'.format(img_name))
                fw.write('{:.1f}\n'.format(dets.shape[0]))
                for k in range(dets.shape[0]):
                    xmin = dets[k, 0]
                    ymin = dets[k, 1]
                    xmax = dets[k, 2]
                    ymax = dets[k, 3]
                    score = dets[k, 4]
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1
                    fw.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(xmin, ymin, w, h, score))
            else:
                for k in range(dets.shape[0]):
                    xmin = dets[k, 0]
                    ymin = dets[k, 1]
                    xmax = dets[k, 2]
                    ymax = dets[k, 3]
                    ymin += 0.2 * (ymax - ymin + 1)
                    score = dets[k, 4]
                    fw.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(img_name, score, xmin, ymin, xmax, ymax))
            print('im_detect: {:d}  misc: {:.4f}s'.format( num_images, _t['misc'].average_time))
    

    fw.close()
