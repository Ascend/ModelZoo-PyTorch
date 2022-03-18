# -*- coding:utf-8 -*-
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
#from pse import pse
import subprocess

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

#if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
#    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))

def pse(kernals, min_area=5):
    '''
    :param kernals:
    :param min_area:
    :return:
    '''
    from pse import pse_cpp
    kernal_num = len(kernals)
    if not kernal_num:
        return np.array([]), []
    kernals = np.array(kernals)
    label_num, label = cv2.connectedComponents(kernals[kernal_num - 1].astype(np.uint8), connectivity=4)
    label_values = []
    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0
            continue
        label_values.append(label_idx)

    pred = pse_cpp(label, kernals, c=7)

    return pred, label_values

image_h = 704
image_w = 1216
ratio_w = 0.95
ratio_h = 0.9777777777777777
img_path = sys.argv[1]
bin_path = sys.argv[2]
txt_path = sys.argv[3]

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    
    for parent, _, filenames in os.walk(img_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    return files

def detect(seg_maps, image_w, image_h, min_area_thresh=10, seg_map_thresh=0.9, ratio = 1):
    '''
    restore text boxes from score map and geo map
    :param seg_maps:
    :param timer:
    :param min_area_thresh:
    :param seg_map_thresh: threshhold for seg map
    :param ratio: compute each seg map thresh
    :return:
    '''
    if len(seg_maps.shape) == 4:
        seg_maps = seg_maps[0, :, :, ]
    #get kernals, sequence: 0->n, max -> min
    kernals = []
    one = np.ones_like(seg_maps[..., 0], dtype=np.uint8)
    zero = np.zeros_like(seg_maps[..., 0], dtype=np.uint8)
    thresh = seg_map_thresh
    for i in range(seg_maps.shape[-1]-1, -1, -1):
        kernal = np.where(seg_maps[..., i]>thresh, one, zero)
        kernals.append(kernal)
        thresh = seg_map_thresh*ratio
    mask_res, label_values = pse(kernals, min_area_thresh)
    mask_res = np.array(mask_res)
    mask_res_resized = cv2.resize(mask_res, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    boxes = []
    for label_value in label_values:
        #(y,x)
        points = np.argwhere(mask_res_resized==label_value)
        points = points[:, (1,0)]
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        boxes.append(box)

    return np.array(boxes), kernals

im_fn_list = get_images()
for im_fn in im_fn_list[8:9]:
    im = cv2.imread(im_fn)[:, :, ::-1]
    idx = os.path.basename(im_fn).split('/')[-1].split('.')[0].split('_')[1]
    seg_maps = np.fromfile(bin_path+"/img_{}_1.bin".format(idx), "float32")
    seg_maps = np.reshape(seg_maps, (1, 7, 176, 304))
    seg_maps = np.transpose(seg_maps, [0, 2, 3, 1])
    print(seg_maps.shape)

    boxes, kernels = detect(seg_maps=seg_maps, image_w=image_w, image_h=image_h)

    if boxes is not None:
        boxes = boxes.reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h
        h, w, _ = im.shape
        boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w)
        boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h)

    # save to file
    if boxes is not None:
        res_file = os.path.join(
            txt_path,
            '{}.txt'.format(os.path.splitext(
                os.path.basename(im_fn))[0]))


        with open(res_file, 'w') as f:
            num =0
            for i in range(len(boxes)):
                box = boxes[i]
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    continue

                num += 1

                f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                    box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]))
                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=2)
    cv2.imshow('result', im)
    cv2.waitKey()