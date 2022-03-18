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

import numpy as np
from PIL import Image, ImageDraw
import os
import multiprocessing
import math

import cfg
from label import shrink


def batch_reorder_vertexes(xy_list_array):
    reorder_xy_list_array = np.zeros_like(xy_list_array)
    for xy_list, i in zip(xy_list_array, range(len(xy_list_array))):
        reorder_xy_list_array[i] = reorder_vertexes(xy_list)
    return reorder_xy_list_array


def reorder_vertexes(xy_list):
    reorder_xy_list = np.zeros_like(xy_list)
    # determine the first point with the smallest x,
    # if two has same x, choose that with smallest y,
    ordered = np.argsort(xy_list, axis=0)
    xmin1_index = ordered[0, 0]
    xmin2_index = ordered[1, 0]
    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]
        first_v = xmin1_index
    # connect the first point to others, the third point on the other side of
    # the line with the middle slope
    #  计算第一个点与其他三个点连线的斜率，取斜率居中的点作为第三个点
    others = list(range(4))
    others.remove(first_v)
    k = np.zeros((len(others),))
    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) \
                    / (xy_list[index, 0] - xy_list[first_v, 0] + cfg.epsilon)
    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]
    # determine the second point which on the bigger side of the middle line
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]  # 得到中间那条线的截距b
    second_v, fourth_v = 0, 0
    for index, i in zip(others, range(len(others))):
        # delta = y - (k * x + b)
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:  # y点在中间那条线之上就设置为第二个点，否则为第四个点
            second_v = index
        else:
            fourth_v = index
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]
    # compare slope of 13 and 24, determine the final order
    #  经过上面的步骤k13有两种情况，要么大于0.要么小于0，k24的斜率正好相反。
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
                xy_list[second_v, 0] - xy_list[fourth_v, 0] + cfg.epsilon)
    if k13 < k24:  # 当k13小于k24的时候，点4变点3，3->2,2->1,1->4
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list


def resize_image(im, max_img_size=cfg.max_train_img_size):  # 把原图长宽根据max_train_img_size变成32的整数倍
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


def gen_npy(o_img_list):
    data_dir = cfg.data_dir
    origin_image_dir = os.path.join(data_dir, cfg.origin_image_dir_name)  # 'image_10000/'
    origin_txt_dir = os.path.join(data_dir, cfg.origin_txt_dir_name)  # 'txt_10000/'
    train_image_dir = os.path.join(data_dir, cfg.train_image_dir_name)  # 'images_%s/' % train_task_id
    train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)  # 'labels_%s/' % train_task_id
    draw_gt_quad = cfg.draw_gt_quad  # True
    show_gt_image_dir = os.path.join(data_dir, cfg.show_gt_image_dir_name)  # 'show_gt_images_%s/' % train_task_id

    for o_img_fname in o_img_list:
        with Image.open(os.path.join(origin_image_dir, o_img_fname)) as im:  # 打开每张图片
            # d_wight, d_height = resize_image(im)
            d_wight, d_height = cfg.max_train_img_size, cfg.max_train_img_size
            scale_ratio_w = d_wight / im.width
            scale_ratio_h = d_height / im.height
            im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')  # 图片缩放
            show_gt_im = im.copy()
            # draw on the img
            draw = ImageDraw.Draw(show_gt_im)
            with open(os.path.join(origin_txt_dir, o_img_fname[:-4] + '.txt'), 'r', encoding='UTF-8') as f:
                anno_list = f.readlines()
            xy_list_array = np.zeros((len(anno_list), 4, 2))
            for anno, i in zip(anno_list, range(len(anno_list))):
                anno_colums = anno.strip().split(',')
                anno_array = np.array(anno_colums)
                xy_list = np.reshape(anno_array[:8].astype(float), (4, 2))
                xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w  # 坐标缩放
                xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
                xy_list = reorder_vertexes(xy_list)  # 坐标顺序转换为统一格式
                xy_list_array[i] = xy_list
                #  将groundtruth文本框内缩，论文为0.3，实际中发现太大，改为0.2
                _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)  # shrink_ratio=0.2，返回长短边都缩后的结果
                shrink_1, _, long_edge = shrink(xy_list, cfg.shrink_side_ratio)  # shrink_side_ratio=0.6，返回仅长边收缩的结果以及长边下标
                if draw_gt_quad:
                    draw.line([tuple(xy_list[0]), tuple(xy_list[1]),
                               tuple(xy_list[2]), tuple(xy_list[3]),
                               tuple(xy_list[0])
                               ],
                              width=2, fill='green')
                    draw.line([tuple(shrink_xy_list[0]),
                               tuple(shrink_xy_list[1]),
                               tuple(shrink_xy_list[2]),
                               tuple(shrink_xy_list[3]),
                               tuple(shrink_xy_list[0])
                               ],
                              width=2, fill='blue')
                    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
                          [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
                    for q_th in range(2):  # 框出头跟尾巴的像素
                        draw.line([tuple(xy_list[vs[long_edge][q_th][0]]),
                                   tuple(shrink_1[vs[long_edge][q_th][1]]),
                                   tuple(shrink_1[vs[long_edge][q_th][2]]),
                                   tuple(xy_list[vs[long_edge][q_th][3]]),
                                   tuple(xy_list[vs[long_edge][q_th][4]])],
                                  width=3, fill='yellow')
            if cfg.gen_origin_img:
                im.save(os.path.join(train_image_dir, o_img_fname))
            np.save(os.path.join(
                train_label_dir,
                o_img_fname[:-4] + '.npy'),
                xy_list_array)  # 保存顺序一致处理后的坐标点集
            if draw_gt_quad:
                show_gt_im.save(os.path.join(show_gt_image_dir, o_img_fname))


def preprocess():
    data_dir = cfg.data_dir
    origin_image_dir = os.path.join(data_dir, cfg.origin_image_dir_name)  # 'image_10000/'
    origin_txt_dir = os.path.join(data_dir, cfg.origin_txt_dir_name)  # 'txt_10000/'
    train_image_dir = os.path.join(data_dir, cfg.train_image_dir_name)  # 'images_%s/' % train_task_id
    train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)  # 'labels_%s/' % train_task_id
    if not os.path.exists(train_image_dir):
        os.mkdir(train_image_dir)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)
    draw_gt_quad = cfg.draw_gt_quad  # True
    show_gt_image_dir = os.path.join(data_dir, cfg.show_gt_image_dir_name)  # 'show_gt_images_%s/' % train_task_id
    if not os.path.exists(show_gt_image_dir):
        os.mkdir(show_gt_image_dir)
    show_act_image_dir = os.path.join(cfg.data_dir, cfg.show_act_image_dir_name)  # 'show_act_images_%s/' % train_task_id
    if not os.path.exists(show_act_image_dir):
        os.mkdir(show_act_image_dir)

    o_img_list = os.listdir(origin_image_dir)
    print('found %d origin images.' % len(o_img_list))
    train_val_set = []
    workers = multiprocessing.cpu_count()
    batch_size = math.ceil(len(o_img_list) / workers)
    batch_list = [o_img_list[i * batch_size:(i + 1) * batch_size] for i in range(workers)]
    thread_pool = multiprocessing.Pool(workers)
    for i in range(workers):
        thread_pool.apply_async(gen_npy, args=(batch_list[i], ))
    thread_pool.close()
    thread_pool.join()
    size = cfg.max_train_img_size
    for o_img_fname in o_img_list:
        train_val_set.append('{},{},{}\n'.format(o_img_fname, size, size))
    train_img_list = os.listdir(train_image_dir)
    print('found %d train images.' % len(train_img_list))
    train_label_list = os.listdir(train_label_dir)
    print('found %d train labels.' % len(train_label_list))

    # random.shuffle(train_val_set)
    train_val_set.sort()
    # 确保每次生成的训练集和验证集一致，不shuffle
    val_count = int(cfg.validation_split_ratio * len(train_val_set))
    with open(os.path.join(data_dir, cfg.val_fname), 'w') as f_val:
        f_val.writelines(train_val_set[:val_count])
    with open(os.path.join(data_dir, cfg.train_fname), 'w') as f_train:
        f_train.writelines(train_val_set[val_count:])


if __name__ == '__main__':
    preprocess()
