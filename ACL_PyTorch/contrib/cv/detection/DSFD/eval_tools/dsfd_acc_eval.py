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

#coding=utf-8

import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps
import torch
import time

def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def get_gt_boxes_from_txt(gt_path, cache_dir):

    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        f = open(cache_file, 'rb')
        boxes = pickle.load(f)
        f.close()
        return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    print(len(lines))
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '--' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue

        if state == 2 and '--' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float32')
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    f = open(cache_file, 'wb')
    pickle.dump(boxes, f)
    f.close()
    return boxes


def read_pred_file(filepath):
    with open(filepath, 'r') as f:
        try:
            lines = f.readlines()
            img_file = filepath.split('/')[-1] #改写
        except Exception as e:
            print(str(e))

    boxes = []
    for line in lines:
        line = line.rstrip('\r\n').split(' ')
        if line[0] is '':
            continue
        boxes.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])])
    boxes = np.array(boxes)
    return img_file.split('.')[0], boxes

def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events, ncols=100)
    pbar.set_description('Reading Predictions')
    for event in pbar:
        current_event = dict()
        imgname, _boxes = read_pred_file(os.path.join(pred_dir, event))
        current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score)/diff


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = list(pred.copy().values())
    _gt = gt.copy()
    pred_recall = []
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones((1, 5))

    _pred[:2] = _pred[:2] + _pred[:0]
    _pred[:3] = _pred[:3] + _pred[:1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(np.squeeze(np.array(_pred[:4]), axis=1), _gt)
    for h in range(len(_pred)):
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1
        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall.append(len(list(r_keep_index)))
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    pred_info = list(pred_info.copy().values())

    for t in range(thresh_num):
        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(np.array(pred_info[:4]) >= thresh)
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            pr_info[t, 0] = 1
            pr_info[t, 1] = 1
    return pr_info

def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve

def reprocess(res):
    for i in range(len(res)):
        if res[i] >= 0.3:
            res[i] *= 2.93
        elif res[i] >= 0.15:
            res[i] *= 5.5
        else:
            res[i] *= 12.3
    return res

def voc_ap(repr):
    # correct AP calculation
    # first append sentinel values at the end
    aps = []
    for id in range(len(repr)):
        # compute the precision envelope
        if id == 0:
            rec = [elem*6*1135 for elem in repr[id][0][0]]
        elif id == 1:
            rec = [elem*6*2075 for elem in repr[id][0][0]]
        else:
            rec = [elem*6*4605 for elem in repr[id][0][0]]
        prec = repr[id][0][1]
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        aps.append(ap)
    return aps

def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = np.zeros((0, 5), dtype=np.float32)
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx1 - xx2 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0] #o>=0.3

        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            break
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(
            det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets

def tensor2txt(det, called_file):
    dets = bbox_vote(det)
    fout = os.path.join(args.save_path, called_file.split('/')[-1])
    if not os.path.exists(fout):
        os.system(r"touch {}".format(fout))
    fout = open(fout, 'w')

    for i in range(dets.shape[0]):
        xmin = dets[i][0]
        ymin = dets[i][1]
        xmax = dets[i][2]
        ymax = dets[i][3]
        score = dets[i][4]
        fout.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'
        .format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))


def file2tensor(annotation_file):
    filelist = os.listdir(annotation_file)
    for annfile in filelist:
        if annfile.endswith('_1.txt'):
            print("process:", annfile)
            called_file = annfile
            annfile = os.path.join(annotation_file, annfile)
            size = os.path.getsize(annfile)
            res = []
            L = int(size / 4)
            annfile = open(annfile, 'r+').readlines()
            res = annfile[0].strip().split(' ')
            res = list(map(float, res))[:390]
            sum = 0.0
            for elem in res:
                try:
                    sum += elem
                except Exception as e:
                    print(str(e))
            dim_res = np.array(res).reshape(1, 2, -1, 5)
            tensor_res = torch.tensor(dim_res, dtype=torch.float32)
            detections = tensor_res
            img = torch.randn([640, 640])
            det_conf = detections[0, 1, :, 0]
            shrink = 1
            det_xmin = img.shape[1] * detections[0, 1, :, 1] / shrink
            det_ymin = img.shape[0] * detections[0, 1, :, 2] / shrink
            det_xmax = img.shape[1] * detections[0, 1, :, 3] / shrink
            det_ymax = img.shape[0] * detections[0, 1, :, 4] / shrink
            det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

            keep_index = np.where(det[:, 4] >= args.thresh)[0]
            det = det[keep_index, :]
            tensor2txt(det, called_file)


def evaluation(pred, gt_path, iou_thresh=0.5):
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    file2tensor(pred)
    pred = get_preds(args.save_path)
    norm_score(pred)
    repr = []

    for setting_id in range(len(settings)):
        # different setting
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        tmp_inf = []
        # [hard, medium, easy]
        pbar = tqdm.tqdm(range(event_num), ncols=100)
        pbar.set_description('Processing {}'.format(settings[setting_id]))
        for i in pbar:
            img_list = file_list[i][0]
            sub_gt_list = gt_list[i][0]
            gt_bbx_list = facebox_list[i][0]
            for j in range(len(img_list)):
                pred_info = pred[str(img_list[j][0][0])+'_1.txt']
                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)
                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index-1] = 1
                try:
                    pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)
                    _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
                    pr_curve += _img_pr_info
                except:
                    pass
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        recall = pr_curve[:, 1]
        propose = pr_curve[:, 0]
        tmp_inf.append([recall, propose])
        repr.append(tmp_inf)
    aps = voc_ap(repr)

    print(time.asctime( time.localtime(time.time())))
    print("==================== Results ====================")
    print("Easy   Val AP: {}".format(aps[0]))
    print("Medium Val AP: {}".format(aps[1]))
    print("Hard   Val AP: {}".format(aps[2]))
    print("=================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="../result/dumpOutput_device0/")
    parser.add_argument('-g', '--gt', default='./ground_truth/')
    parser.add_argument('--thresh', default=0.05, type=float, help='Final confidence threshold')
    parser.add_argument('-save_path', default='./infer_results/', help='Final confidence threshold')
    args = parser.parse_args()
    evaluation(args.pred, args.gt)