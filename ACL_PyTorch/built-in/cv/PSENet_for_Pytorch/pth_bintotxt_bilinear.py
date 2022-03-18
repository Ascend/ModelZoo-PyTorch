import os
import sys
import numpy as np
import torch
import cv2
from pypse import pse as pypse
import torch.nn.functional as F


img_path = sys.argv[1]
bin_path = sys.argv[2]
txt_path = sys.argv[3]

if not os.path.exists(txt_path):
    os.makedirs(txt_path)

kernel_num=7
min_kernel_area=5.0
scale=1
min_score = 0.9
min_area = 600


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


im_fn_list = get_images()
for im_fn in im_fn_list:
    print(im_fn)
    im = cv2.imread(im_fn)
    idx = os.path.basename(im_fn).split('/')[-1].split('.')[0].split('_')[1]
    seg_maps = np.fromfile(bin_path+"/img_{}_1.bin".format(idx), "float32")
    seg_maps = np.reshape(seg_maps, (1, 7, 176, 304))
    seg_maps = torch.from_numpy(seg_maps)

    # Resize 算子
    seg_maps = F.interpolate(seg_maps, size=(704, 1216), mode='bilinear', align_corners=False)
    # print(seg_maps)
    # print(seg_maps.shape)
    #
    # seg_maps = seg_maps.float()
    score = torch.sigmoid(seg_maps[:, 0, :, :])
    outputs = (torch.sign(seg_maps - 1.0) + 1) / 2

    text = outputs[:, 0, :, :]
    kernels = outputs[:, 0:kernel_num, :, :] * text

    score = score.data.numpy()[0].astype(np.float32)
    text = text.data.numpy()[0].astype(np.uint8)
    kernels = kernels.numpy()[0].astype(np.uint8)

    # python version pse
    pred = pypse(kernels, min_kernel_area / (scale * scale))

    img_scale = (im.shape[1] * 1.0 / pred.shape[1], im.shape[0] * 1.0 / pred.shape[0])
    label = pred
    label_num = np.max(label) + 1
    bboxes = []

    for i in range(1, label_num):
        points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

        if points.shape[0] < min_area:
            continue

        score_i = np.mean(score[label == i])
        if score_i < min_score:
            continue

        rect = cv2.minAreaRect(points)
        bbox = cv2.boxPoints(rect) * img_scale
        bbox = bbox.astype('int32')
        bboxes.append(bbox.reshape(-1))
    # print(bboxes)
    # save txt
    res_file = os.path.join(txt_path,'{}.txt'.format(os.path.splitext(os.path.basename(im_fn))[0]))
    with open(res_file, 'w') as f:
        for b_idx, bbox in enumerate(bboxes):
            values = [int(v) for v in bbox]
            line = "%d, %d, %d, %d, %d, %d, %d, %d\n" % tuple(values)
            f.write(line)


    # show result
    # for bbox in bboxes:
    #     cv2.drawContours(im, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    # cv2.imshow('result', im)
    # cv2.waitKey()


