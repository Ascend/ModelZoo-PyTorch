#!/usr/bin/env python
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
from math import sqrt, isnan, log, ceil
import cv2


class Heatmapper:
    """
    It can generate the keypoint heatmaps, body part heatmap, offset feature maps in development.
    """
    # 杈撳叆鍥剧墖鐨勫昂瀵稿鐞
    # 璁粌鏃堕渶瑕佺浉鍚屽ぇ灏忕殑鍥剧墖鎵嶈兘缁勬垚涓涓猙atch锛屽湪openpose涓湁涓ょ鍋氭硶锛
    # 涓鏄洿鎺esize鍒版寚瀹氬ぇ灏忕殑灏哄;
    # 浜屾槸婧愮爜鎻愪緵浜嗕竴绉嶇◢寰湁鐗硅壊鐨勫仛娉:銆
    # 鍏堟寚瀹氶暱鍜屽x锛寉銆傜劧鍚庡皢鍥剧墖鐨勯暱 / 瀹藉拰x / y姣旇緝锛岀湅鏄惁澶т簬1
    # 鐒跺悗锛岄夋嫨闀夸竴浜涚殑杈癸紙闀 > x?, 瀹 > y?)锛屽浐瀹氶暱瀹芥瘮缂╂斁鍒扮粰瀹氬昂瀵
    # 鍐嶇劧鍚庯紝涓哄彟涓鏉¤竟鍔爌adding锛屼袱杈瑰姞鐩稿悓鐨刾adding
    # 鏈鍚庯紝resize鍒版寚瀹氬ぇ灏忋

    def __init__(self, config):

        self.config = config
        self.sigma = config.transform_params.sigma
        self.paf_sigma = config.transform_params.paf_sigma
        self.double_sigma2 = 2 * self.sigma * self.sigma
        # set responses lower than gaussian_thre to 0
        self.keypoint_gaussian_thre = config.transform_params.keypoint_gaussian_thre
        self.limb_gaussian_thre = config.transform_params.limb_gaussian_thre
        self.gaussian_size = ceil((sqrt(-self.double_sigma2 * log(self.keypoint_gaussian_thre))) / config.stride) * 2
        self.offset_size = self.gaussian_size // 2 + 1  # + 1  # offset vector range
        self.thre = config.transform_params.paf_thre

        # cached common parameters which same for all iterations and all pictures
        stride = self.config.stride
        width = self.config.width // stride
        height = self.config.height // stride

        # x, y coordinates of centers of bigger grid, stride / 2 -0.5鏄负浜嗗湪璁＄畻鍝嶅簲鍥炬椂锛屼娇鐢╣rid鐨勪腑蹇
        self.grid_x = np.arange(width) * stride + stride / 2 - 0.5  # x -> width
        self.grid_y = np.arange(height) * stride + stride / 2 - 0.5  # y -> height

        # x ,y indexes (type: int) of heatmap feature maps
        self.Y, self.X = np.mgrid[0:self.config.height:stride, 0:self.config.width:stride]
        # 瀵<numpy.lib.index_tricks.MGridClass object> slice鎿嶄綔锛屾瘮濡侺[:10:2]鍓10涓暟锛屾瘡闅斾袱涓彇涓涓
        # # basically we should use center of grid, but in this place classic implementation uses left-top point.
        self.X = self.X + stride / 2 - 0.5
        self.Y = self.Y + stride / 2 - 0.5

    def create_heatmaps(self, joints,  mask_all):  # 鍥惧儚鏍规嵁姣忎釜main person閮借澶勭悊鎴愪簡鍥哄畾鐨勫ぇ灏忓昂瀵革紝鍥犳heatmap涔熸槸鍥哄畾澶у皬浜
        """
        Create keypoint and body part heatmaps
        :param joints: input keypoint coordinates, np.float32 dtype is a very little faster
        :param mask_miss: mask areas without keypoint annotation
        :param mask_all: all person (including crowd) area mask (denoted as 1)
        :return: Masked groundtruth heatmaps!
        """
        # print(joints.shape)  # 渚嬪(3, 18, 3)锛屾妸姣忎釜main person浣滀负鍥剧墖鐨勪腑蹇冿紝浣嗘槸渚濈劧鍙兘浼氬寘鎷叾浠栦笉鍚岀殑浜哄湪杩欎釜瑁佸壀鍚庣殑鍥惧儚涓
        heatmaps = np.zeros(self.config.parts_shape, dtype=np.float32)  # config.parts_shape: 46, 46, 57
        # 姝ゅ鐨刪eat map涓鍏辨湁57涓猚hannel锛屽寘鍚簡heat map浠ュ強paf浠ュ強鑳屾櫙channel銆
        # 骞朵笖瀵筯eat map鍒濆鍖栦负0寰堥噸瑕侊紝鍥犱负杩欐牱浣垮緱娌℃湁鏍囨敞鐨勫尯鍩熸槸娌℃湁鍊肩殑锛
        self.put_joints(heatmaps, joints)
        # sl = slice(self.config.heat_start, self.config.heat_start + self.config.heat_layers)
        # python鍒囩墖鍑芥暟銆class slice(start, stop[, step])
        # Generate foreground of keypoint heat map 鍒犻櫎浜嗕竴浜涗唬鐮侊紝鍘熷璇峰弬鑰冧箣鍓嶅body part椤圭洰
        # heatmaps[:, :, self.config.bkg_start] = 1. - np.amax(heatmaps[:, :, sl], axis=2)

        # # 鏌愪釜浣嶇疆鐨勮儗鏅痟eatmap鍊煎畾涔変负杩欎釜鍧愭爣浣嶇疆澶勩鏈澶х殑鏌愪釜绫诲瀷鑺傜偣楂樻柉鍝嶅簲鐨勮ˉ 1. - np.amax(heatmaps[:, :, sl], axis=2)
        # 濡傛灉鍔犲叆鐨勬槸鍓嶆櫙鑰屼笉鏄儗鏅紝鍒欏搷搴旀槸銆np.amax(heatmaps[:, :, sl], axis=2)

        self.put_limbs(heatmaps, joints)

        # add foreground (mask_all) channel, i.e., the person segmentation mask
        kernel = np.ones((3, 3), np.uint8)
        mask_all = cv2.erode(mask_all, kernel)  # crop the boundary of mask_all
        heatmaps[:, :, self.config.bkg_start] = mask_all

        # add reverse keypoint gaussian heat map on the second background channel
        sl = slice(self.config.heat_start, self.config.heat_start + self.config.heat_layers)  # consider all real joints
        heatmaps[:, :, self.config.bkg_start + 1] = np.amax(heatmaps[:, :, sl], axis=2)  # 1 -  #鍘熸潵鏄彇鍙嶇殑

        # 閲嶈锛佷笉瑕佸繕浜嗗皢鐢熸垚鐨刧roundtruth heatmap涔樹互mask锛屼互姝ゆ帺鐩栨帀娌℃湁鏍囨敞鐨刢rowd浠ュ強鍙湁寰堝皯keypoint鐨勪汉
        # 骞朵笖锛岃儗鏅殑mask_all娌℃湁涔樹互mask_miss锛岃缁冩椂鍙槸瀵规病鏈夊叧閿偣鏍囨敞鐨刪eatmap鍖哄煙mask鎺変笉鍋氱洃鐫ｏ紝鑰屼笉闇瑕佸杈撳叆鍥剧墖mask!
        # heatmaps *= mask_miss[:, :, np.newaxis]  # fixme: 鏀惧湪loss璁＄畻涓紝瀵筸ask_all涓嶉渶瑕佷箻mask_miss锛屼笉缂烘爣娉

        # see: https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/124
        # Mask never touch pictures.  Mask涓嶄細鍙犲姞鍒癷mage鏁版嵁涓
        # Mask has exactly same dimensions as ground truth and network output. ie 46 x 46 x num_layers.
        # ------------------------------------------------------------- #
        # Mask applied to:
        # * ground truth heatmap and pafs (multiplied by mask)
        # * network output (multiplied by mask)
        # ------------------------------------------------------------- #
        # If in same point of answer mask is zero this means "ignore answers in this point while training network"
        # because loss will be zero in this point.
        heatmaps = np.clip(heatmaps, 0., 1.)  # 闃叉鏁版嵁寮傚父
        return heatmaps.transpose((2, 0, 1))  # pytorch need N*C*H*W format

    def put_gaussian_maps(self, heatmaps, layer, joints):
        # update: 鍙绠椾竴瀹氬尯鍩熷唴鑰屼笉鏄叏鍥惧儚鐨勫兼潵鍔犻烥T鐨勭敓鎴愶紝鍙傝僡ssociate embedding
        #  change the gaussian map to laplace map to get a shapper peak of keypoint ?? the result is not good
        # actually exp(a+b) = exp(a)*exp(b), lets use it calculating 2d exponent, it could just be calculated by

        for i in range(joints.shape[0]):  # 澶栧眰寰幆鏄姣忎竴涓猨oint閮藉湪瀵瑰簲绫诲瀷channel鐨刦eature map涓婁骇鐢熶竴涓珮鏂垎甯

            # --------------------------------------------------------------------------------------------------#
            # 杩欓噷鏄釜鎶宸э紝grid_x鍏跺疄鍙栧艰寖鍥存槸0~368锛岃捣鐐规槸3.5锛岀粓鐐瑰兼槸363.5锛岄棿闅斾负8锛岃繖鏍峰氨鏄湪鍘熷368涓綅缃笂璁＄畻楂樻柉鍊硷紝
            # 閲囨牱浜46涓偣锛屼粠鑰屾渶澶х▼搴︿繚鐣欎簡鍘熷鍒嗚鲸鐜囧昂瀵镐笂鐨勫搷搴斿硷紝閬垮厤閲忓寲璇樊锛佽屼笉鏄敓鎴愬師濮嬪垎杈ㄧ巼澶у皬鐨刧round truth鐒跺悗缂╁皬8鍊嶃銆

            # 濡傛灉浣跨敤楂樻柉鍒嗗竷锛岄檺鍒秅uassin response鐢熸垚鐨勫尯鍩燂紝浠ユ鍔犲揩杩愮畻
            x_min = int(round(joints[i, 0] / self.config.stride) - self.gaussian_size // 2)
            x_max = int(round(joints[i, 0] / self.config.stride) + self.gaussian_size // 2 + 1)
            y_min = int(round(joints[i, 1] / self.config.stride) - self.gaussian_size // 2)
            y_max = int(round(joints[i, 1] / self.config.stride) + self.gaussian_size // 2 + 1)

            if y_max < 0:
                continue

            if x_max < 0:
                continue

            if x_min < 0:
                x_min = 0

            if y_min < 0:
                y_min = 0

            # this slice is not only speed up but crops the keypoints off the transformed picture really
            # slice can also crop the extended index of a numpy array and return empty array []
            slice_x = slice(x_min, x_max)
            slice_y = slice(y_min, y_max)

            exp_x = np.exp(-(self.grid_x[slice_x].astype(np.float32) - joints[i, 0]) ** 2 /
                           np.array([self.double_sigma2]).astype(np.float32))
            exp_y = np.exp(-(self.grid_y[slice_y].astype(np.float32) - joints[i, 1]) ** 2 /
                           np.array([self.double_sigma2]).astype(np.float32))

            exp = np.outer(exp_y, exp_x)  # np.outer鐨勮绠楋紝涓や釜闀垮害涓篗,N鐨勫悜閲忕殑澶栫Н缁撴灉鏄疢*N鐨勭煩闃
            # --------------------------------------------------------------------------------------------------#

            # # heatmap銆濡傛灉浣跨敤鎷夋櫘鎷夋柉鍒嗗竷锛歞is = exp-(math.sqrt((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma)
            # dist = np.sqrt((self.X - joints[i, 0])**2 + (self.Y - joints[i, 1])**2) / 2.0 / self.sigma
            # np.where(dist > 4.6052, 1e8, dist) # 璺濈涓績澶繙鐨勪笉璧嬪
            # exp = np.exp(-dist)

            # note this is correct way of combination - min(sum(...),1.0) as was in C++ code is incorrect
            # https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/118
            heatmaps[slice_y, slice_x, self.config.heat_start + layer] = \
                np.maximum(heatmaps[slice_y, slice_x, self.config.heat_start + layer], exp)
            # 杩欎竴鍙ヤ唬鐮佹槸瑙ｅ喅濡備綍澶勭悊涓涓綅缃彲鑳芥湁涓嶅悓浜虹殑鍏宠妭鐐圭殑楂樻柉鍝嶅簲鍥剧殑鐢熸垚鈥滆鐩栤濈殑闂锛屼笉鍙栬繖涓や釜鐐圭殑楂樻柉鍒嗗竷鐨勫钩鍧囷紝鑰屾槸鍙栨渶澶у
            # Notice!
            # ------------------------------------------------------------------------------------------------ #
            # 姣忎竴鏉℃洸绾跨殑宄板奸兘琛ㄧず杩欎釜浣嶇疆瀛樺湪鍏抽敭鐐圭殑鍙兘鎬ф渶楂橈紝濡傝鏂囧叕寮(7)鍥炬墍绀猴紝鍙兘鏈変袱涓叧閿偣璺濈姣旇緝杩戯紝杩欎袱鏉￠珮鏂洸绾
            # 濡傛灉鍙栧钩鍧囧肩殑璇濓紝寰堟槑鏄惧氨浠庝袱涓嘲鍊煎彉鎴愪竴涓嘲鍊间簡锛岄偅鏈鍚庨娴嬪嚭鐨勭粨鏋滃彲鑳藉氨鍙湁涓涓叧閿偣浜嗐傛墍浠ヨ繖閲屽彇鐨勬槸鏈澶у笺
            # ------------------------------------------------------------------------------------------------ #

    def put_joints(self, heatmaps, joints):

        for i in range(self.config.num_parts):  # len(config.num_parts) = 18, 涓嶅寘鎷儗鏅痥eypoint
            visible = joints[:, i, 2] < 2  # only annotated (visible) keypoints are considered !
            self.put_gaussian_maps(heatmaps, i, joints[visible, i, 0:2])  # 閫愪釜channel鍦拌繘琛実round truth鐨勭敓鎴

    def put_limb_gaussian_maps(self, heatmaps, layer, joint_from, joint_to):
        """
        鐢熸垚涓涓猚hannel涓婄殑PAF groundtruth
        """

        count = np.zeros(heatmaps.shape[:-1], dtype=np.float32)  # count鐢ㄦ潵璁板綍鏌愪竴涓綅缃偣涓婃湁澶氬皯闈為浂鐨刾af锛屼互渚垮悗闈㈠仛骞冲潎
        for i in range(joint_from.shape[0]):
            (x1, y1) = joint_from[i]
            (x2, y2) = joint_to[i]

            dx = x2 - x1
            dy = y2 - y1
            dnorm = dx * dx + dy * dy

            if dnorm == 0:  # we get nan here sometimes, it's kills NN
                # we get nan here sometimes, it's kills NN
                # handle it better. probably we should add zero paf, centered paf,
                # or skip this completely. add a special paf?
                # 鎴戣涓哄彲浠ヤ笉鐢ㄥ幓澶勭悊锛屽湪鍚庡鐞嗘椂锛屾妸娌℃湁褰㈡垚limb鐨勭偣鍒嗛厤缁欒窛绂绘渶杩戠殑閭ｄ釜浜哄嵆鍙
                print("Parts are too close to each other. Length is zero. Skipping")
                continue

            dx = dx / dnorm
            dy = dy / dnorm

            assert not isnan(dx) and not isnan(dy), "dnorm is zero, wtf"

            min_sx, max_sx = (x1, x2) if x1 < x2 else (x2, x1)
            min_sy, max_sy = (y1, y2) if y1 < y2 else (y2, y1)

            # include the two end-points of the limbs
            min_sx = int(round((min_sx - self.thre) / self.config.stride))
            min_sy = int(round((min_sy - self.thre) / self.config.stride))
            max_sx = int(round((max_sx + self.thre) / self.config.stride))
            max_sy = int(round((max_sy + self.thre) / self.config.stride))

            # check whether PAF is off screen. do not really need to do it with max>grid size
            if max_sy < 0:
                continue

            if max_sx < 0:
                continue

            if min_sx < 0:
                min_sx = 0

            if min_sy < 0:
                min_sy = 0

            # this slice mask is not only speed up but crops paf really. This copied from original code
            # max_sx + 1, array slice dose not include the last element.
            # could be wrong if the keypint locates at the edge of the image but this seems never to happen.
            slice_x = slice(min_sx, max_sx + 1)
            slice_y = slice(min_sy, max_sy + 1)
            # tt = self.X[slice_y,slice_x]
            dist = distances(self.X[slice_y, slice_x], self.Y[slice_y, slice_x], self.paf_sigma, x1, y1, x2, y2,
                             self.limb_gaussian_thre)
            # 杩欓噷姹傜殑璺濈鏄湪鍘熷灏哄368*368鐨勫昂瀵革紝鑰屼笉鏄缉灏8鍊嶅悗鍦46*46涓婄殑璺濈锛岀劧鍚庢斁鍒46*46鍒囩墖slice鐨勪綅缃笂鍘
            # print(dist.shape)
            heatmaps[slice_y, slice_x, layer][dist > 0] += dist[dist > 0]  # = dist * dx銆鑻ヤ笉鍋氬钩鍧囷紝鍒欎笉杩涜绱姞

            count[slice_y, slice_x][dist > 0] += 1

        #  averaging by pafs mentioned in the paper but never worked in C++ augmentation code 鎴戦噰鐢ㄤ簡骞冲潎
        heatmaps[:, :, layer][count > 0] /= count[count > 0]  # 杩欎簺閮芥槸鐭㈤噺鍖栵紙鐭╅樀锛夋搷浣

    def put_limbs(self, heatmaps, joints):
        """
         # 寰幆璋冪敤閫愪釜channel鐢熸垚ground truth鐨勫嚱鏁帮紝鏈澶栧眰寰幆鏄搴旀煇涓猯imb鐨勬煇涓涓猚hannel
        """
        for (i, (fr, to)) in enumerate(self.config.limbs_conn):
            visible_from = joints[:, fr, 2] < 2  # 鍒ゆ柇璇ョ偣鏄惁琚爣娉ㄤ簡
            visible_to = joints[:, to, 2] < 2
            visible = visible_from & visible_to  # &: 鎸変綅鍙朼nd, 鍙湁涓や釜鑺傜偣閮芥爣娉ㄤ簡鎵嶈兘鐢熸垚paf, v=0,1鏃惰〃绀鸿鐐硅鏍囨敞浜
            # In this project:  0 - marked but invisible, 1 - marked and visible, which is different from coco銆dataset

            layer = self.config.paf_start + i
            self.put_limb_gaussian_maps(heatmaps, layer, joints[visible, fr, 0:2], joints[visible, to, 0:2])

    def put_offset_vector_maps(self, offset_vectors, mask_offset, layer, joints):
        """
        鐢熸垚offset heatmap
        :param offset_vectors:
        :param mask_offset:
        :param layer: 灏嗗綋鍓峯ffset鏀剧疆鍦 2*layer, 2*layer+1 channel涓
        :param joints:
        :return:
        """
        for i in range(joints.shape[0]):
            x_min = int(round(joints[i, 0] / self.config.stride) - self.offset_size // 2)
            x_max = int(round(joints[i, 0] / self.config.stride) + self.offset_size // 2 + 1)
            y_min = int(round(joints[i, 1] / self.config.stride) - self.offset_size // 2)
            y_max = int(round(joints[i, 1] / self.config.stride) + self.offset_size // 2 + 1)

            if y_max < 0:
                continue

            if x_max < 0:
                continue

            if x_min < 0:
                x_min = 0

            if y_min < 0:
                y_min = 0

            # this slice is not only speed up but crops the keypoints off the transformed picture really
            # slice can also crop the extended index of a numpy array and return empty array []
            slice_x = slice(x_min, x_max)
            slice_y = slice(y_min, y_max)

            # Try: 灏唎ffset鐢╨og鍑芥暟缂栫爜涓嶅悎閫傦紝鍥犱负鈭唜, 鈭唝鏈夋鏈夎礋銆傚彲浠ュ厛灏嗗亸宸紪鐮佸埌-0.5锝0.5锛屽啀浣跨敤L1 loss
            # type: np.ndarray # joints[i, 0] -> x
            offset_x = (self.grid_x[slice_x].astype(np.float32) - joints[i, 0]) / (self.offset_size * self.config.stride)
            # type: np.ndarray # joints[i, 1] -> y
            offset_y = (self.grid_y[slice_y].astype(np.float32) - joints[i, 1]) / (self.offset_size * self.config.stride)
            offset_x_mesh = np.repeat(offset_x.reshape(1, -1), offset_y.shape[0], axis=0)
            offset_y_mesh = np.repeat(offset_y.reshape(-1, 1), offset_x.shape[0], axis=1)

            offset_vectors[slice_y, slice_x, layer * 2] += offset_x_mesh  # add up the offsets in the same location
            offset_vectors[slice_y, slice_x, layer * 2 + 1] += offset_y_mesh
            mask_offset[slice_y, slice_x, layer * 2] += 1
            mask_offset[slice_y, slice_x, layer * 2 + 1] += 1

    def put_offset(self, joints):
        offset_vectors = np.zeros(self.config.offset_shape, dtype=np.float32)
        mask_offset = np.zeros(self.config.offset_shape, dtype=np.float32)
        assert offset_vectors.shape[-1] == 2 * self.config.num_parts, 'offset map depth dose not match keypoint number'

        for i in range(self.config.num_parts):  # len(config.num_parts) = 18, 涓嶅寘鎷儗鏅痥eypoint
            visible = joints[:, i, 2] < 2  # only annotated (visible) keypoints are considered !
            self.put_offset_vector_maps(offset_vectors, mask_offset, 0, joints[visible, i, 0:2])  # 鎵鏈夊叧閿偣鍏变韩offset channel

        offset_vectors[mask_offset > 0] /= mask_offset[mask_offset > 0]  # average the offsets in the same location
        mask_offset[mask_offset > 0] = 1  # reset the offset mask area

        return offset_vectors.transpose((2, 0, 1)), mask_offset.transpose((2, 0, 1))  # pytorch need N*C*H*W format


def gaussian(sigma, x, u):

    double_sigma2 = 2 * sigma ** 2
    y = np.exp(- (x - u) ** 2 / double_sigma2)
    return y


def distances(X, Y, sigma, x1, y1, x2, y2, thresh=0.01, return_dist=False):  # TODO: change the paf area to ellipse
    """
    杩欓噷鐨刣istance鍑芥暟瀹為檯涓婅繑鍥炵殑鏄痝auss鍒嗗竷鐨凱AF
    # 鐐瑰埌涓や釜绔偣鎵纭畾鐨勭洿绾跨殑璺濈銆classic formula is:
    # # d = [(x2-x1)*(y1-y)-(x1-x)*(y2-y1)] / sqrt((x2-x1)**2 + (y2-y1)**2)
    """
    # parallel_encoding calculation distance from any number of points of arbitrary shape(X, Y),
    # to line defined by segment (x1,y1) -> (x2, y2)
    xD = (x2 - x1)
    yD = (y2 - y1)
    detaX = x1 - X
    detaY = y1 - Y
    norm2 = sqrt(xD ** 2 + yD ** 2)  # 娉ㄦ剰norm2鏄竴涓暟鑰屼笉鏄痭umpy鏁扮粍,鍥犱负xD, yD閮芥槸涓涓暟銆傚崟涓暟瀛楄繍绠梞ath姣攏umpy蹇
    dist = xD * detaY - detaX * yD  # 甯告暟涓巒umpy鏁扮粍(X,Y鏄潗鏍囨暟缁,澶氫釜鍧愭爣锛夌殑杩愮畻锛宐roadcast
    dist /= (norm2 + 1e-6)
    dist = np.abs(dist)
    if return_dist:
        return dist
    # ratiox = np.abs(detaX / (xD + 1e-8))
    # ratioy = np.abs(detaY / (yD + 1e-8))
    # ratio = np.where(ratiox < ratioy, ratiox, ratioy)
    # ratio = np.where(ratio > 1, 1, ratio)  # 涓嶇敤銆np.ones_like(ratio)涔熷彲浠ユ甯歌繍琛岋紝骞朵笖浼氬揩涓鐐圭偣
    # ratio = np.where(ratio > 0.5, 1 - ratio, ratio)
    # oncurve_dist = b * np.sqrt(1 - np.square(ratio * 2))  # oncurve_dist璁＄畻鐨勬槸妞渾杈圭晫涓婄殑鐐瑰埌闀胯酱鐨勫瀭鐩磋窛绂

    guass_dist = gaussian(sigma, dist, 0)
    # TODO: 涓嬩竴涓崲鎴# =0.01
    guass_dist[guass_dist <= thresh] = 0.01   # thresh  # 0.67鐨凩2鐢ㄧ殑鏄0  # 0.68 flocal 鐢ㄧ殑=thresh 鍚屽墠闈㈢殑鍏抽敭鐐瑰搷搴旓紝澶繙鐨勪笉瑕
    # b = thre
    # guass_dist[dist >= b] = 0

    return guass_dist


def test():
    hm = Heatmapper()
    d = distances(hm.X, hm.Y, 100, 100, 50, 150)
    print(d < 8.)


if __name__ == "__main__":
    np.set_printoptions(precision=1, linewidth=1000, suppress=True, threshold=100000)
    test()
