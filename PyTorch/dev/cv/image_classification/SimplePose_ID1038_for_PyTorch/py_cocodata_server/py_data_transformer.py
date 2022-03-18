# coding:utf-8
#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
import numpy as np
from math import cos, sin, pi
import cv2
import random
import matplotlib.pyplot as plt


class AugmentSelection:
    def __init__(self, flip=False, tint=False, degree=0., crop=(0, 0), scale=1.):
        self.flip = flip
        self.tint = tint
        self.degree = degree  # rotate
        self.crop = crop  # shift actually
        self.scale = scale

    @staticmethod  # staticmethod鏀寔绫诲璞℃垨鑰呭疄渚嬪鏂规硶鐨勮皟鐢
    def random(transform_params):
        flip = random.uniform(0., 1.) < transform_params.flip_prob
        tint = random.uniform(0., 1.) < transform_params.tint_prob
        degree = random.uniform(-1., 1.) * transform_params.max_rotate_degree

        scale = (transform_params.scale_max - transform_params.scale_min) * random.uniform(0., 1.) + \
                transform_params.scale_min \
            if random.uniform(0., 1.) < transform_params.scale_prob else 1.

        x_offset = int(random.uniform(-1., 1.) * transform_params.center_perterb_max)
        y_offset = int(random.uniform(-1., 1.) * transform_params.center_perterb_max)

        return AugmentSelection(flip, tint, degree, (x_offset, y_offset), scale)

    @staticmethod
    def unrandom():
        flip = False
        tint = False
        degree = 0.
        scale = 1.
        x_offset = 0
        y_offset = 0

        return AugmentSelection(flip, tint, degree, (x_offset, y_offset), scale)

    def affine(self, center, scale_self, config):
        # the main idea: we will do all image transformations with one affine matrix.
        # this saves lot of cpu and make code significantly shorter
        # same affine matrix could be used to transform joint coordinates afterwards
        scale_self *= (config.height / (config.height - 1))

        A = cos(self.degree / 180. * pi)
        B = sin(self.degree / 180. * pi)

        scale_size = config.transform_params.target_dist / scale_self * self.scale
        # target_dist鏄皟鏁翠汉鍗犳暣涓浘鍍忕殑姣斾緥鍚楋紵
        # It used in picture augmentation during training. Rough meaning is "height of main person on image should
        # be approximately 0.6 of the original image size". It used in this file in my code:
        # https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation/blob/master/py_rmpe_server/py_rmpe_transformer.py
        # This mean we will scale picture so height of person always will be 0.6 of picture.
        # After it we apply random scaling (self.scale) from 0.6 to 1.1
        (width, height) = center
        center_x = width
        center_y = height

        # 涓轰簡澶勭悊鏂逛究锛屽皢鍥惧儚鍙樻崲鍒颁互鍘熺偣涓轰腑蹇
        center2zero = np.array([[1., 0., -center_x],
                                [0., 1., -center_y],
                                [0., 0., 1.]])

        rotate = np.array([[A, B, 0],
                           [-B, A, 0],
                           [0, 0, 1.]])

        scale = np.array([[scale_size, 0, 0],
                          [0, scale_size, 0],
                          [0, 0, 1.]])

        flip = np.array([[-1 if self.flip else 1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])

        # 鏈鍚庡啀浠庡師鐐逛腑蹇冨彉鎹㈠埌鎸囧畾鍥惧儚澶у皬灏哄鐨勪腑蹇冧笂鍘诲苟涓旇繘琛岄殢鏈哄钩绉
        center2center = np.array([[1., 0., config.width / 2 - 0.5 + self.crop[0]],
                                  [0., 1., config.height / 2 - 0.5 + self.crop[1]],
                                  [0., 0., 1.]])

        # order of combination is reversed
        # 杩欏彇鍐充簬鍧愭爣鏄鍚戦噺杩樻槸鍒楀悜閲忥紝瀵瑰簲鍙樻崲鐭╅樀鏄乏涔樿繕鏄彸涔橈紝姝ゅ鍧愭爣鐢ㄧ殑鏄垪鍚戦噺褰㈠紡
        combined = center2center.dot(flip).dot(scale).dot(rotate).dot(center2zero)

        return combined[0:2], scale_size


class Transformer:
    def __init__(self, config):

        self.config = config

    @staticmethod  # staticmethod鏀寔绫诲璞℃垨鑰呯被鐨勫疄渚嬪鏂规硶鐨勮皟鐢
    def distort_color(img):
        # uint8 input锛宱pencv outputs Hue銆丼aturation銆乂alue ranges are: [0,180)锛孾0,256)锛孾0,256)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv_img[:, :, 0] = np.maximum(np.minimum(hsv_img[:, :, 0] - 10 + np.random.randint(20 + 1), 179),
                                      0)  # hue
        hsv_img[:, :, 1] = np.maximum(np.minimum(hsv_img[:, :, 1] - 20 + np.random.randint(80 + 1), 255),
                                      0)  # saturation
        hsv_img[:, :, 2] = np.maximum(np.minimum(hsv_img[:, :, 2] - 20 + np.random.randint(60 + 1), 255),
                                      0)  # value
        hsv_img = hsv_img.astype(np.uint8)

        distorted_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return distorted_img

    def transform(self, img, mask_miss, mask_all, meta, aug=None):
        """ If aug is None, then do random augmentation. Input original data and output transformed data """

        if aug is None:
            aug = AugmentSelection.random(self.config.transform_params)

        if aug.tint:
            img = self.distort_color(img)
        # # ------------------------------------------------------------------------------------ #

        # warp picture and mask
        assert meta['scale_provided'][0] != 0, "************ scale_proviede is zero, dividing zero! ***********"

        M, scale_size = aug.affine(meta['objpos'][0], meta['scale_provided'][0], self.config)
        # 鏍规嵁鎺掑悕绗竴鐨刴ain person杩涜鍥惧儚缂╂斁
        # need to understand this,
        # scale_provided[0] is height of main person divided by 512, calculated in generate_hdf5.py
        # print(img.shape)

        # 鍙樻崲涔嬪悗杩樹細缂╂斁鍒癱onfig.height澶у皬, (self.config.height, self.config.width)銆鎸囧畾鐨勬槸杩斿洖鍥惧儚鐨勫昂瀵
        img = cv2.warpAffine(img, M, (self.config.height, self.config.width), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(124, 127, 127))
        # for debug, see the transformed data
        # plt.imshow(img[:,:,[2,1,0]])  # opencv imread ---> BGR order
        # plt.show()

        # mask涔熻鍋氫竴鑷寸殑鍙樻崲  FIXME: resize鎻掑肩畻娉曟敼鎴愪笁娆＄珛鏂
        mask_miss = cv2.warpAffine(mask_miss, M, (self.config.height, self.config.width), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)  # cv2.INTER_CUBIC閫傚悎鏀惧ぇ

        mask_miss = cv2.resize(mask_miss, self.config.mask_shape,     # mask shape銆鏄粺涓鐨 46*46
                          interpolation=cv2.INTER_AREA)

        mask_all = cv2.warpAffine(mask_all, M, (self.config.height, self.config.width), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        #
        mask_all = cv2.resize(mask_all, self.config.mask_shape,    # mask shape銆鏄粺涓鐨 46*46
                          interpolation=cv2.INTER_AREA)

        # # debug usage: show the image and corresponding mask area
        # # mask areas are in dark when display
        # plt.imshow(img[:, :, [2, 1, 0]])
        # plt.imshow(np.repeat(mask_image_size[:, :, np.newaxis], 3, axis=2), alpha=0.5)  # mask_all
        # plt.show()

        # warp key points
        # Issue: joint could be cropped by augmentation, in this case we should mark it as invisible.
        # update: may be we don't need it actually, original code removed part sliced more than half totally,
        # may be we should keep it
        original_points = meta['joints'].copy()
        original_points[:, :, 2] = 1  # we reuse 3rd column in completely different way here, it is hack
        # -----------------------------------------------------------------------------銆#
        # 闇瑕佹坊鍔犺秴杩囪竟鐣屾椂姝ゆ椂璁句负2鍚楋紵 涓婇潰鐨剈pdate宸茬粡鍥炵瓟浜嗚繖涓棶棰橈紝鍦╤eatmaper.py鐢熸垚鏃朵娇鐢ㄤ簡slice
        # -----------------------------------------------------------------------------銆#

        # we use 0 - hidden, 1 visible, 2 absent - it is not coco values they processed by generate_hdf5
        converted_points = np.matmul(M, original_points.transpose([0, 2, 1])).transpose([0, 2, 1])  # 鍏抽敭鐐圭殑鍧愭爣鍙樻崲
        # 浠庣煩闃电浉涔樼殑鏂瑰紡鏉ョ湅锛屽潗鏍囪〃绀虹敤鐨勬槸鍒楀悜閲忥紝鎵浠ユ槸宸︿箻鍙樻崲鐭╅樀
        meta['joints'][:, :, 0:2] = converted_points

        # we just made image flip, i.e. right leg just became left leg, and vice versa
        if aug.flip:
            tmpLeft = meta['joints'][:, self.config.leftParts, :]  # 閫氳繃涓棿鐨勭紦瀛樺彉閲忚繘琛屼氦鎹
            tmpRight = meta['joints'][:, self.config.rightParts, :]
            meta['joints'][:, self.config.leftParts, :] = tmpRight
            meta['joints'][:, self.config.rightParts, :] = tmpLeft
        # print('*********************', img.shape, meta['joints'].shape)
        # meta['joints'].shape = (num_of_person, 18, 3)锛屽叾涓18鏄18涓叧閿偣锛3浠ｈ〃锛坸,y,v)

        # normalize image to 0~1 here to save gpu/cpu time
        # mask - 闄や互255涔嬪悗锛岃mask鍦版柟鏄0.0,娌℃湁mask鍦版柟鏄1.0
        # return transformed data as flot32 format
        return img.astype(np.float32)/255., mask_miss.astype(np.float32)/255., mask_all.astype(np.float32)/255., meta

