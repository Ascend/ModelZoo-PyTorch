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
"""
Hint: please ingore the chinease annotations whcih may be wrong and they are just remains from old version.
"""

import sys
import json
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import tqdm
import time
import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.config_reader import config_reader
from utils import util
from config.config import GetConfig, COCOSourceConfig, TrainingOpt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from models.posenet import NetworkEval
import warnings
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # choose the available GPUs
warnings.filterwarnings("ignore")

# For visualize
colors = [[128, 114, 250], [130, 238, 238], [48, 167, 238], [180, 105, 255], [255, 0, 0], [255, 85, 0], [255, 170, 0],
          [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
          [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170],
          [255, 0, 85], [193, 193, 255], [106, 106, 255], [20, 147, 255]]

torch.npu.empty_cache()
parser = argparse.ArgumentParser(description='PoseNet Training')
parser.add_argument('--resume', '-r', action='store_true', default=True, help='resume from checkpoint')
parser.add_argument('--max_grad_norm', default=5, type=float,
                    help="If the norm of the gradient vector exceeds this, re-normalize it to have the norm equal to max_grad_norm")
parser.add_argument('--image', type=str, default='try_image/ski.jpg', help='input image')  # required=True
parser.add_argument('--output', type=str, default='result.jpg', help='output image')
parser.add_argument('--opt-level', type=str, default='O1')
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

args = parser.parse_args()

# ###################################  Setup for some configurations ###########################################
opt = TrainingOpt()
config = GetConfig(opt.config_name)

limbSeq = config.limbs_conn
dt_gt_mapping = config.dt_gt_mapping
flip_heat_ord = config.flip_heat_ord
flip_paf_ord = config.flip_paf_ord
draw_list = config.draw_list


# ###############################################################################################################


def show_color_vector(oriImg, paf_avg, heatmap_avg):
    hsv = np.zeros_like(oriImg)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(paf_avg[:, :, 16], 1.5 * paf_avg[:, :, 16])  # 璁剧疆涓嶅悓鐨勭郴鏁帮紝鍙互浣垮緱鏄剧ず棰滆壊涓嶅悓

    # 灏嗗姬搴﹁浆鎹负瑙掑害锛屽悓鏃禣penCV涓殑H鑼冨洿鏄180(0 - 179)锛屾墍浠ュ啀闄や互2
    # 瀹屾垚鍚庡皢缁撴灉璧嬬粰HSV鐨凥閫氶亾锛屼笉鍚岀殑瑙掑害(鏂瑰悜)浠ヤ笉鍚岄鑹茶〃绀
    # 瀵逛簬涓嶅悓鏂瑰悜锛屼骇鐢熶笉鍚岃壊璋
    # hsv[...,0]绛変环浜巋sv[:,:,0]
    hsv[..., 0] = ang * 180 / np.pi / 2

    # 灏嗙煝閲忓ぇ灏忔爣鍑嗗寲鍒0-255鑼冨洿銆傚洜涓篛penCV涓璙鍒嗛噺瀵瑰簲鐨勫彇鍊艰寖鍥存槸256
    # 瀵逛簬鍚屼竴H銆丼鑰岃█锛屽悜閲忕殑澶у皬瓒婂ぇ锛屽搴旈鑹茶秺浜
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # 鏈鍚庯紝灏嗙敓鎴愬ソ鐨凥SV鍥惧儚杞崲涓築GR棰滆壊绌洪棿
    limb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    plt.imshow(oriImg[:, :, [2, 1, 0]])
    plt.imshow(limb_flow, alpha=.5)
    plt.show()

    plt.imshow(oriImg[:, :, [2, 1, 0]])
    plt.imshow(paf_avg[:, :, 11], alpha=.6)
    plt.show()

    plt.imshow(heatmap_avg[:, :, -1])
    plt.imshow(oriImg[:, :, [2, 1, 0]], alpha=0.25)  # show a keypoint
    plt.show()

    plt.imshow(heatmap_avg[:, :, -2])
    plt.imshow(oriImg[:, :, [2, 1, 0]], alpha=0.5)  # show the person mask
    plt.show()

    plt.imshow(oriImg[:, :, [2, 1, 0]])  # show a keypoint
    plt.imshow(heatmap_avg[:, :, 4], alpha=.5)
    plt.show()
    t = 2


def process(input_image, params, model_params, heat_layers, paf_layers):
    oriImg = cv2.imread(input_image)  # B,G,R order.    璁粌鏁版嵁鐨勮鍏ヤ篃鏄敤opencv锛屽洜姝や篃鏄疊, G, R椤哄簭
    # oriImg = cv2.resize(oriImg, (768, 768))
    # oriImg = cv2.flip(oriImg, 1) 鍥犱负璁粌鏃朵綔浜唂lip锛屾墍浠ョ敤杩欑鏂瑰紡鎻愬崌骞舵病鏈変綔鐢
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]  # 鎸夌収鍥剧墖楂樺害杩涜缂╂斁
    # multipier = [0.21749408983451538, 0.43498817966903075, 0.6524822695035462, 0.8699763593380615],
    # 棣栧厛鎶婅緭鍏ュ浘鍍忛珮搴﹀彉鎴368,鐒跺悗鍐嶅仛缂╂斁

    heatmap_avg = np.zeros(
        (oriImg.shape[0], oriImg.shape[1], heat_layers))  # fixme if you change the number of keypoints
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], paf_layers))

    for m in range(len(multiplier)):
        scale = multiplier[m]

        if scale * oriImg.shape[0] > 2300 or scale * oriImg.shape[1] > 3200:
            scale = min(2300 / oriImg.shape[0], 3200 / oriImg.shape[1])
            print("Input image is too big, shrink it !")

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)  # cv2.INTER_CUBIC
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['max_downsample'],
                                                          model_params['padValue'])

        # ################################# Important!  ###########################################
        # #############################  We use OpenCV to read image (BGR) all the time #######################
        # Input Tensor: a batch of images within [0,1], required shape in this project : (1, height, width, channels)
        input_img = np.float32(imageToTest_padded / 255)
        # input_img -= np.array(config.img_mean[::-1])  # Notice: OpenCV uses BGR format, reverse the last axises
        # input_img /= np.array(config.img_std[::-1])
        # ################################## add flip image ################################
        swap_image = input_img[:, ::-1, :].copy()
        # plt.imshow(swap_image[:, :, [2, 1, 0]])  # Opencv image format: BGR
        # plt.show()
        input_img = np.concatenate((input_img[None, ...], swap_image[None, ...]),
                                   axis=0)  # (2, height, width, channels)
        input_img = torch.from_numpy(input_img).npu()
        # ###################################################################################

        # output tensor dtype: float 16
        output_tuple = posenet(input_img)

        # ############ different scales can be shown #############
        output = output_tuple[-1][0].cpu().numpy()

        output_blob = output[0].transpose((1, 2, 0))
        output_blob0 = output_blob[:, :, :config.paf_layers]
        output_blob1 = output_blob[:, :, config.paf_layers:config.num_layers]

        output_blob_flip = output[1].transpose((1, 2, 0))
        output_blob0_flip = output_blob_flip[:, :, :config.paf_layers]  # paf layers
        output_blob1_flip = output_blob_flip[:, :, config.paf_layers:config.num_layers]  # keypoint layers

        # ################################## flip ensemble ################################
        output_blob0_avg = (output_blob0 + output_blob0_flip[:, ::-1, :][:, :, flip_paf_ord]) / 2
        output_blob1_avg = (output_blob1 + output_blob1_flip[:, ::-1, :][:, :, flip_heat_ord]) / 2

        # extract outputs, resize, and remove padding
        heatmap = cv2.resize(output_blob1_avg, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        # output_blob0 is PAFs
        paf = cv2.resize(output_blob0_avg, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        # ##############################     涓轰簡璁╁钩鍧噃eatmap涓嶉偅涔堟ā绯婏紵     ################################3
        # heatmap[heatmap < params['thre1']] = 0
        # paf[paf < params['thre2']] = 0
        # ####################################################################################### #

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

        heatmap_avg[np.isnan(heatmap_avg)] = 0
        paf_avg[np.isnan(paf_avg)] = 0

        # heatmap_avg = np.maximum(heatmap_avg, heatmap)
        # paf_avg = np.maximum(paf_avg, paf)  # 濡傛灉鎹㈡垚鍙栨渶澶э紝鏁堟灉浼氬彉宸紝鏈夊緢澶氳妫

    all_peaks = []
    peak_counter = 0
    # --------------------------------------------------------------------------------------- #
    # ------------------------  show the limb and foreground channel  -----------------------#
    # --------------------------------------------------------------------------------------- #

    show_color_vector(oriImg, paf_avg, heatmap_avg)

    # --------------------------------------------------------------------------------------- #
    # ####################################################################################### #
    # ------------------------- find keypoints  ---------------------------------------------#
    # ####################################################################################### #
    # --------------------------------------------------------------------------------------- #

    # smoothing = util.GaussianSmoothing(18, 5, 1)
    # heatmap_avg_npu = torch.from_numpy(heatmap_avg.transpose((2, 0, 1))).npu()[None, ...]

    heatmap_avg = heatmap_avg.astype(np.float32)

    filter_map = heatmap_avg[:, :, :18].copy().transpose((2, 0, 1))[None, ...]
    filter_map = torch.from_numpy(filter_map).npu()

    # # #######################   Add Gaussian smooth  #######################
    # smoothing = util.GaussianSmoothing(18, 7, 1)
    # filter_map = F.pad(filter_map, (3, 3, 3, 3), mode='reflect')
    # filter_map = smoothing(filter_map)
    # # ######################################################################

    filter_map = util.keypoint_heatmap_nms(filter_map, kernel=3, thre=params['thre1'])
    filter_map = filter_map.cpu().numpy().squeeze().transpose((1, 2, 0))

    for part in range(18):  # 娌℃湁瀵硅儗鏅紙搴忓彿19锛夊彇闈炴瀬澶у兼姂鍒禢MS
        map_ori = heatmap_avg[:, :, part]
        # map = gaussian_filter(map_ori, sigma=3)  # 娌℃湁楂樻柉婊ゆ尝璨屼技鏁堟灉鏇村ソ锛
        # map = map_ori
        # map up 鏄
        peaks_binary = filter_map[:, :, part]

        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
        # note reverse. xy鍧愭爣绯诲拰鍥惧儚鍧愭爣绯
        # np.nonzero: Return the indices of the elements that are non-zero
        # 娣诲姞鍔犳潈鍧愭爣璁＄畻锛屾牴鎹笉鍚岀被鍨嬪叧閿偣寮ユ暎绋嬪害涓嶅悓閫夋嫨鍔犳潈鐨勮寖鍥
        refined_peaks_with_score = [util.refine_centroid(map_ori, anchor, params['offset_radius']) for anchor in peaks]

        # peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]  # 鍒楄〃瑙ｆ瀽寮忥紝鐢熶骇鐨勬槸list  # refined_peaks
        # [(205, 484, 0.9319216758012772),
        #  # (595, 484, 0.777797631919384),
        id = range(peak_counter, peak_counter + len(refined_peaks_with_score))
        peaks_with_score_and_id = [refined_peaks_with_score[i] + (id[i],) for i in range(len(id))]
        # 涓烘瘡涓涓浉搴攑eak (parts)閮戒緷娆＄紪浜嗕竴涓彿

        all_peaks.append(peaks_with_score_and_id)
        # all_peaks.append 濡傛灉姝ょ鍏宠妭绫诲瀷娌℃湁鍏冪礌锛宎ppend涓涓┖鐨刲ist []锛屼緥濡俛ll_peaks[19]:
        # [(205, 484, 0.9319216758012772, 25),
        # (595, 484, 0.777797631919384, 26),
        # (343, 490, 0.8145177364349365, 27), ....
        peak_counter += len(peaks)  # refined_peaks

    # --------------------------------------------------------------------------------------- #
    # ####################################################################################### #
    # ----------------------------- find connections -----------------------------------------#
    # ####################################################################################### #
    # --------------------------------------------------------------------------------------- #

    connection_all = []
    special_k = []

    # 鏈夊灏戜釜limb,灏辨湁澶氬皯涓猚onnection,鐩稿搴斿湴灏辨湁澶氬皯涓猵af channel
    for k in range(len(limbSeq)):  # 鏈澶栧眰鐨勫惊鐜槸鏌愪竴涓猯imbSeq
        score_mid = paf_avg[:, :, k]  # 鏌愪竴涓猚hannel涓妉imb鐨勫搷搴旂儹鍥, 瀹冪殑闀垮涓庡師濮嬭緭鍏ュ浘鐗囧ぇ灏忎竴鑷达紝鍓嶉潰缁忚繃resize浜
        # score_mid = gaussian_filter(orginal_score_mid, sigma=3)  fixme: use gaussisan blure?
        candA = all_peaks[limbSeq[k][0]]  # all_peaks鏄痩ist,姣忎竴琛屼篃鏄竴涓猯ist,淇濆瓨浜嗘娴嬪埌鐨勭壒瀹氱殑parts(joints)
        # 娉ㄦ剰鍏蜂綋澶勭悊鏃舵爣鍙蜂粠0杩樻槸1寮濮嬨備粠鏀堕泦鐨刾eaks涓彇鍑烘煇绫诲叧閿偣锛坧art)闆嗗悎
        candB = all_peaks[limbSeq[k][1]]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    mid_num = min(int(round(norm + 1)), params['mid_num'])
                    # failure case when 2 body parts overlaps
                    if norm == 0:  # 涓轰簡璺宠繃鍑虹幇涓嶅悓鑺傜偣鐩镐簰瑕嗙洊鍑虹幇鍦ㄥ悓涓涓綅缃紝涔熸湁璇磏orm鍔犱竴涓帴杩0鐨勯」閬垮厤鍒嗘瘝涓0,璇﹁锛
                        # https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/54
                        continue

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    limb_response = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0]))] \
                                              for I in range(len(startend))])

                    score_midpts = limb_response

                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    # 杩欎竴椤规槸涓轰簡鎯╃綒杩囬暱鐨刢onnection, 鍙湁褰撻暱搴﹀ぇ浜庡浘鍍忛珮搴︾殑涓鍗婃椂鎵嶄細鎯╃綒 todo
                    # The term of sum(score_midpts)/len(score_midpts), see the link below.
                    # https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/issues/48

                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > params['connect_ration'] * len(
                        score_midpts)  # fixme: tune 鎵嬪姩璋冩暣, 鏈潵鏄 > 0.8*len
                    # 鎴戣涓鸿繖涓垽鍒爣鍑嗘槸淇濊瘉paf鏈濆悜鐨勪竴鑷存  param['thre2']
                    # parm['thre2'] = 0.05
                    criterion2 = score_with_dist_prior > 0

                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, norm,
                                                     0.5 * score_with_dist_prior + 0.25 * candA[i][2] + 0.25 * candB[j][
                                                         2]])
                        # todo:鐩存帴鎶婁袱绉嶇被鍨嬫鐜囩浉鍔犱笉鍚堢悊
                        # connection_candidate鎺掑簭鐨勪緷鎹槸dist prior姒傜巼鍜屼袱涓鐐筯eat map棰勬祴鐨勬鐜囧
                        # How to undersatand the criterion?

            connection_candidate = sorted(connection_candidate, key=lambda x: x[4], reverse=True)
            # sorted 鍑芥暟瀵瑰彲杩唬瀵硅薄锛屾寜鐓ey鍙傛暟鎸囧畾鐨勫璞¤繘琛屾帓搴忥紝revers=True鏄寜鐓ч嗗簭鎺掑簭锛宻ort涔嬪悗鍙互鎶婃渶鍙兘鏄痩imb鐨勭暀涓嬶紝鑰屾妸鍜屾渶鍙兘鏄痩imb鐨勭鐐圭珵浜夌殑绔偣鍒犻櫎

            connection = np.zeros((0, 6))
            for c in range(len(connection_candidate)):  # 鏍规嵁confidence鐨勯『搴忛夋嫨connections
                i, j, s, limb_len = connection_candidate[c][0:4]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    # 杩涜鍒ゆ柇纭繚涓嶄細鍑虹幇涓や釜绔偣闆嗗悎A,B涓紝鍑虹幇涓涓泦鍚堜腑鐨勭偣涓庡彟澶栦竴涓泦鍚堜腑涓や釜鐐瑰悓鏃剁浉杩
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j, limb_len]])  # 鍚庨潰浼氳浣跨敤
                    # candA[i][3], candB[j][3]鏄痯art鐨刬d缂栧彿
                    if (len(connection) >= min(nA, nB)):  # 浼氬嚭鐜板叧鑺傜偣涓嶅杩炵殑鎯呭喌
                        break
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
            # 涓涓┖鐨刐]涔熻兘鍔犲叆鍒發ist涓紝杩欎竴鍙ユ槸蹇呴』鐨勶紒鍥犱负connection_all鐨勬暟鎹粨鏋勬槸姣忎竴琛屼唬琛ㄤ竴绫籰imb connection

    # --------------------------------------------------------------------------------------- #
    # ####################################################################################### #
    # --------------------------------- find people ------------------------------------------#
    # ####################################################################################### #
    # --------------------------------------------------------------------------------------- #

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20, 2))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    # candidate[:, 2] *= 0.5  # FIXME: change it? part confidence * 0.5
    # candidate.shape = (94, 4). 鍒楄〃瑙ｆ瀽寮忥紝涓ゅ眰寰幆锛屽厛浠巃ll peaks鍙栵紝鍐嶄粠sublist涓彇銆 all peaks鏄袱灞俵ist

    for k in range(len(limbSeq)):
        # ---------------------------------------------------------
        # 澶栧眰寰幆limb  瀵瑰簲璁烘枃涓紝姣忎竴涓猯imb灏辨槸涓涓瓙闆嗭紝鍒唋imb澶勭悊,璐績绛栫暐?
        # special_K ,琛ㄧず娌℃湁鎵惧埌鍏宠妭鐐瑰鍖归厤鐨勮偄浣
        if k not in special_k:  # 鍗炽鏈変笌涔嬬浉杩炵殑锛岃繖涓猵af(limb)鏄瓨鍦ㄧ殑
            partAs = connection_all[k][:, 0]  # limb绔偣part鐨勫簭鍙凤紝涔熷氨鏄繚瀛樺湪candidate涓殑  id鍙
            partBs = connection_all[k][:, 1]  # limb绔偣part鐨勫簭鍙凤紝涔熷氨鏄繚瀛樺湪candidate涓殑  id鍙
            # connection_all 姣忎竴琛屾槸涓涓被鍨嬬殑limb,姣忎竴琛屾牸寮: N * [idA, idB, score, i, j]
            indexA, indexB = np.array(limbSeq[k])  # 姝ゆ椂澶勭悊limb k,limbSeq鐨勪袱涓鐐筽arts锛屾槸parts鐨勭被鍒彿.
            #  鏍规嵁limbSeq鍒楄〃鐨勯『搴忎緷娆¤冨療鏌愮绫诲瀷鐨刲imb锛屼粠涓涓叧鑺傜偣鍒颁笅涓涓叧鑺傜偣

            for i in range(len(connection_all[k])):  # 璇ュ眰寰幆鏄垎閰峩绫诲瀷鐨刲imb connection銆(partAs[i],partBs[i])鍒版煇涓汉銆subset[]
                # ------------------------------------------------
                # 姣忎竴琛岀殑list淇濆瓨鐨勬槸涓绫籰imb(connection),閬嶅巻鎵鏈夋绫籰imb,涓鑸殑鏈夊灏戜釜鐗瑰畾鐨刲imb灏辨湁澶氬皯涓汉

                found = 0
                subset_idx = [-1, -1]  # 姣忔寰幆鍙В鍐充袱涓猵art锛屾墍浠ユ爣璁板彧闇瑕佷袱涓猣lag
                for j in range(len(subset)):
                    # ----------------------------------------------
                    # 杩欎竴灞傚惊鐜槸閬嶅巻鎵鏈夌殑浜

                    # 1:size(subset,1), 鑻ubset.shape=(5,20), 鍒檒en(subset)=5锛岃〃绀烘湁5涓汉
                    # subset姣忎竴琛屽搴旂殑鏄竴涓汉鐨18涓叧閿偣鍜宯umber浠ュ強score鐨勭粨鏋
                    if subset[j][indexA][0].astype(int) == (partAs[i]).astype(int) or subset[j][indexB][0].astype(
                            int) == partBs[i].astype(int):
                        # 鐪嬬湅杩欐鑰冨療鐨刲imb涓や釜绔偣涔嬩竴鏄惁鏈変竴涓凡缁忓湪涓婁竴杞腑鍑虹幇杩囦簡,鍗虫槸鍚﹀凡缁忓垎閰嶇粰鏌愪汉浜
                        # 姣忎竴涓渶澶栧眰寰幆閮藉彧鑰冭檻涓涓猯imb锛屽洜姝ゅ鐞嗙殑鏃跺欏氨鍙細鏈変袱绉峱art,鍗宠〃绀轰负partAs,partBs
                        subset_idx[found] = j  # 鏍囪涓涓嬶紝杩欎釜绔偣搴旇鏄j涓汉鐨
                        found += 1

                if found == 1:
                    j = subset_idx[0]

                    if subset[j][indexB][0].astype(int) == -1 and \
                            params['len_rate'] * subset[j][-1][1] > connection_all[k][i][-1]:
                        # 濡傛灉鏂板姞鍏ョ殑limb姣斾箣鍓嶅凡缁忕粍瑁呯殑limb闀垮緢澶氾紝涔熻垗寮
                        # 濡傛灉杩欎釜浜虹殑褰撳墠鐐硅繕娌℃湁琚壘鍒版椂锛屾妸杩欎釜鐐瑰垎閰嶇粰杩欎釜浜
                        # 杩欎竴涓垽鏂潪甯搁噸瑕侊紝鍥犱负绗18鍜19涓猯imb鍒嗗埆鏄 2->16, 5->17,杩欏嚑涓偣宸茬粡鍦ㄤ箣鍓嶇殑limb涓娴嬪埌浜嗭紝
                        # 鎵浠ュ鏋滀袱娆＄粨鏋滀竴鑷达紝涓嶆洿鏀规鏃剁殑part鍒嗛厤锛屽惁鍒欏張鍒嗛厤浜嗕竴娆★紝缂栧彿鏄鐩栦簡锛屼絾鏄户缁繍琛屼笅闈唬鐮侊紝part鏁扮洰
                        # 浼氬姞锛戯紝缁撴灉閫犳垚涓涓汉鐨刾art涔嬪拰>18銆備笉杩囧鏋滀袱渚ч娴媗imb绔偣缁撴灉涓嶅悓锛岃繕鏄細鍑虹幇number of part>18锛岄犳垚澶氭
                        # FIXME: 娌℃湁鍒╃敤濂藉啑浣欑殑connection淇℃伅锛屾渶鍚庝袱涓猯imb鐨勭鐐逛笌涔嬪墠寰幆杩囩▼涓噸澶嶄簡锛屼絾娌℃湁鍒╃敤鑱氬悎锛
                        #  鍙槸鐩存帴瑕嗙洊锛屽叾瀹炵洿鎺ヨ鐩栨槸涓轰簡寮ヨˉ婕忔

                        subset[j][indexB][0] = partBs[i]  # partBs[i]鏄痩imb鍏朵腑涓涓鐐圭殑id鍙风爜
                        subset[j][indexB][1] = connection_all[k][i][2]  # 淇濆瓨杩欎釜鐐硅鐣欎笅鏉ョ殑缃俊搴
                        subset[j][-1][0] += 1  # last number in each row is the total parts number of that person

                        # # subset[j][-2][1]鐢ㄦ潵璁板綍涓嶅寘鎷綋鍓嶆柊鍔犲叆鐨勭被鍨嬭妭鐐规椂鐨勬讳綋鍒濆缃俊搴︼紝寮曞叆瀹冩槸涓轰簡閬垮厤涓嬫杩唬鍑虹幇鍚岀被鍨嬪叧閿偣锛岃鐩栨椂閲嶅鐩稿姞浜嗙疆淇″害
                        # subset[j][-2][1] = subset[j][-2][0]  # 鍥犱负鏄笉鍖呮嫭姝ょ被鑺傜偣鐨勫垵濮嬪硷紝鎵浠ュ彧浼氳祴鍊间竴娆 !!

                        subset[j][-2][0] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                        # candidate鐨勬牸寮忎负锛  (343, 490, 0.8145177364349365, 27), ....
                        subset[j][-1][1] = max(connection_all[k][i][-1], subset[j][-1][1])

                        # the second last number in each row is the score of the overall configuration

                    elif subset[j][indexB][0].astype(int) != partBs[i].astype(int):
                        if subset[j][indexB][1] >= connection_all[k][i][2]:
                            # 濡傛灉鑰冨療鐨勮繖涓猯imb杩炴帴娌℃湁宸茬粡瀛樺湪鐨勫彲淇★紝鍒欒烦杩
                            pass

                        else:
                            # 鍚﹀垯鐢ㄥ綋鍓嶇殑limb绔偣瑕嗙洊宸茬粡瀛樺湪鐨勭偣锛屽苟涓斿湪杩欎箣鍓嶏紝鍑忓幓宸插瓨鍦ㄥ叧鑺傜偣鐨勭疆淇″害鍜岃繛鎺ュ畠鐨刲imb缃俊搴
                            if params['len_rate'] * subset[j][-1][1] <= connection_all[k][i][-1]:
                                continue
                            # 鍑忓幓涔嬪墠鐨勮妭鐐圭疆淇″害鍜宭imb缃俊搴
                            subset[j][-2][0] -= candidate[subset[j][indexB][0].astype(int), 2] + subset[j][indexB][1]

                            # 娣诲姞褰撳墠鑺傜偣
                            subset[j][indexB][0] = partBs[i]
                            subset[j][indexB][1] = connection_all[k][i][2]  # 淇濆瓨杩欎釜鐐硅鐣欎笅鏉ョ殑缃俊搴
                            subset[j][-2][0] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                            subset[j][-1][1] = max(connection_all[k][i][-1], subset[j][-1][1])

                    #  overlap the reassigned keypoint
                    #  濡傛灉鏄坊鍔犲啑浣欒繛鎺ョ殑閲嶅鐨勭偣锛岀敤鏂扮殑鏇村姞楂樼殑鍐椾綑杩炴帴姒傜巼鍙栦唬鍘熸潵杩炴帴鐨勭浉鍚岀殑鍏宠妭鐐圭殑姒傜巼
                    # 杩欎竴涓敼鍔ㄦ病鍟ュ奖鍝
                    elif subset[j][indexB][0].astype(int) == partBs[i].astype(int) and subset[j][indexB][1] <= \
                            connection_all[k][i][2]:
                        # 鍚﹀垯鐢ㄥ綋鍓嶇殑limb绔偣瑕嗙洊宸茬粡瀛樺湪鐨勭偣锛屽苟涓斿湪杩欎箣鍓嶏紝鍑忓幓宸插瓨鍦ㄥ叧鑺傜偣鐨勭疆淇″害鍜岃繛鎺ュ畠鐨刲imb缃俊搴
                        if params['len_rate'] * subset[j][-1][1] <= connection_all[k][i][-1]:
                            continue
                        # 鍑忓幓涔嬪墠鐨勮妭鐐圭疆淇″害鍜宭imb缃俊搴
                        subset[j][-2][0] -= candidate[subset[j][indexB][0].astype(int), 2] + subset[j][indexB][1]

                        # 娣诲姞褰撳墠鑺傜偣
                        subset[j][indexB][0] = partBs[i]
                        subset[j][indexB][1] = connection_all[k][i][2]  # 淇濆瓨杩欎釜鐐硅鐣欎笅鏉ョ殑缃俊搴
                        subset[j][-2][0] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                        subset[j][-1][1] = max(connection_all[k][i][-1], subset[j][-1][1])

                elif found == 2:  # if found 2 and disjoint, merge them (disjoint锛氫笉鐩镐氦)
                    # -----------------------------------------------------
                    # 濡傛灉鑲綋缁勬垚鐨勫叧鑺傜偣A,B鍒嗗埆杩炲埌浜嗕袱涓汉浣擄紝鍒欒〃鏄庤繖涓や釜浜轰綋搴旇缁勬垚涓涓汉浣擄紝
                    # 鍒欏悎骞朵袱涓汉浣擄紙褰撹偄浣撴槸鎸夐『搴忔嫾鎺ユ儏鍐典笅涓嶅瓨鍦ㄨ繖鏍风殑鐘跺喌锛
                    # --------------------------------------------------

                    # 璇存槑缁勮鐨勮繃绋嬩腑锛屾湁鏂帀鐨勬儏鍐碉紙鏈塴imb鎴栬呰connection缂哄け锛夛紝鍦ㄤ箣鍓嶉噸澶嶅紑杈熶簡涓涓猻ub person,鍏跺疄浠栦滑鏄悓涓涓汉涓婄殑
                    # If humans H1 and H2 share a part index with the same coordinates, they are sharing the same part!
                    #  H1 and H2 are, therefore, the same humans. So we merge both sets into H1 and remove H2.
                    # https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-2-e78ab9104fc8
                    # 璇ヤ唬鐮佷笌閾炬帴涓殑鍋氭硶鏈夊樊寮傦紝涓汉璁や负閾炬帴涓殑鏇村姞鍚堢悊鑰屼笖鏇村鏄撶悊瑙
                    j1, j2 = subset_idx

                    membership1 = ((subset[j1][..., 0] >= 0).astype(int))[:-2]  # 鐢╗:,0]涔熷彲
                    membership2 = ((subset[j2][..., 0] >= 0).astype(int))[:-2]
                    membership = membership1 + membership2
                    # [:-2]涓嶅寘鎷渶鍚庝釜鏁伴」涓巗cores椤
                    # 杩欎簺鐐瑰簲璇ュ睘浜庡悓涓涓汉,灏嗚繖涓汉鎵鏈夌被鍨嬪叧閿偣锛堢鐐筽art)涓暟閫愪釜鐩稿姞
                    if len(np.nonzero(membership == 2)[0]) == 0:  # if found 2 and disjoint, merge them

                        min_limb1 = np.min(subset[j1, :-2, 1][membership1 == 1])
                        min_limb2 = np.min(subset[j2, :-2, 1][membership2 == 1])
                        min_tolerance = min(min_limb1, min_limb2)  # 璁＄畻鍏佽杩涜鎷兼帴鐨勭疆淇″害

                        if connection_all[k][i][2] < params['connection_tole'] * min_tolerance or params['len_rate'] * \
                                subset[j1][-1][1] <= connection_all[k][i][-1]:
                            # 濡傛灉merge杩欎袱涓韩浣撻儴鍒嗙殑缃俊搴︿笉澶熷ぇ锛屾垨鑰呭綋鍓嶈繖涓猯imb鏄庢樉澶т簬宸插瓨鍦ㄧ殑limb鐨勯暱搴︼紝鍒欎笉杩涜杩炴帴
                            # todo: finetune the tolerance of connection
                            continue  #

                        subset[j1][:-2][...] += (subset[j2][:-2][...] + 1)
                        # 瀵逛簬娌℃湁鑺傜偣鏍囪鐨勫湴鏂癸紝鍥犱负涓よsubset鐩稿簲浣嶇疆澶勯兘鏄-1,鎵浠ュ悎骞朵箣鍚庢病鏈夎妭鐐圭殑閮ㄥ垎渚濇棫鏄-锛
                        # 鎶婁笉鐩镐氦鐨勪袱涓猻ubset[j1],[j2]涓殑id鍙疯繘琛岀浉鍔狅紝浠庤屽畬鎴愬悎骞讹紝杩欓噷+1鏄洜涓洪粯璁ゆ病鏈夋壘鍒板叧閿偣鍒濆鍊兼槸-1

                        subset[j1][-2:][:, 0] += subset[j2][-2:][:, 0]  # 涓よsubset鐨勭偣鐨勪釜鏁板拰鎬荤疆淇″害鐩稿姞

                        subset[j1][-2][0] += connection_all[k][i][2]
                        subset[j1][-1][1] = max(connection_all[k][i][-1], subset[j1][-1][1])
                        # 娉ㄦ剰锛氥鍥犱负鏄痙isjoint鐨勪袱琛宻ubset鐐圭殑merge锛屽洜姝ゅ厛鍓嶅瓨鍦ㄧ殑鑺傜偣鐨勭疆淇″害涔嬪墠宸茬粡琚姞杩囦簡 !! 杩欓噷鍙渶瑕佸啀鍔犲綋鍓嶈冨療鐨刲imb鐨勭疆淇″害
                        subset = np.delete(subset, j2, 0)

                    else:
                        # 鍑虹幇浜嗕袱涓汉鍚屾椂绔炰簤涓涓猯imb鐨勬儏鍐碉紝骞朵笖杩欎袱涓汉涓嶆槸鍚屼竴涓汉锛岄氳繃姣旇緝涓や釜浜哄寘鍚limb鐨勭疆淇″害鏉ュ喅瀹氾紝
                        # 褰撳墠limb鐨勮妭鐐瑰簲璇ュ垎閰嶇粰璋侊紝鍚屾椂鎶婁箣鍓嶇殑閭ｄ釜涓庡綋鍓嶈妭鐐圭浉杩炵殑鑺傜偣(鍗硃artsA[i])浠庡彟涓涓汉(subset)鐨勮妭鐐归泦鍚堜腑鍒犻櫎
                        if connection_all[k][i][0] in subset[j1, :-2, 0]:
                            c1 = np.where(subset[j1, :-2, 0] == connection_all[k][i][0])
                            c2 = np.where(subset[j2, :-2, 0] == connection_all[k][i][1])
                        else:
                            c1 = np.where(subset[j1, :-2, 0] == connection_all[k][i][1])
                            c2 = np.where(subset[j2, :-2, 0] == connection_all[k][i][0])

                        # c1, c2鍒嗗埆鏄綋鍓峫imb杩炴帴鍒癹1浜虹殑绗琧1涓叧鑺傜偣锛宩2浜虹殑绗琧2涓叧鑺傜偣
                        c1 = int(c1[0])
                        c2 = int(c2[0])
                        assert c1 != c2, "an candidate keypoint is used twice, shared by two people"

                        # 濡傛灉褰撳墠鑰冨療鐨刲imb缃俊搴︽瘮宸茬粡瀛樺湪鐨勪袱涓汉杩炴帴鐨勭疆淇″害灏忥紝鍒欒烦杩囷紝鍚﹀垯鍒犻櫎宸插瓨鍦ㄧ殑涓嶅彲淇＄殑杩炴帴鑺傜偣銆
                        if connection_all[k][i][2] < subset[j1][c1][1] and connection_all[k][i][2] < subset[j2][c2][1]:
                            continue  # the trick here is useful

                        small_j = j1
                        big_j = j2
                        remove_c = c1

                        if subset[j1][c1][1] > subset[j2][c2][1]:
                            small_j = j2
                            big_j = j1
                            remove_c = c2
                        # 鍒犻櫎鍜屽綋鍓峫imb鏈夎繛鎺,骞朵笖缃俊搴︿綆鐨勯偅涓汉鐨勮妭鐐
                        if params['remove_recon'] > 0:
                            subset[small_j][-2][0] -= candidate[subset[small_j][remove_c][0].astype(int), 2] + \
                                                      subset[small_j][remove_c][1]
                            subset[small_j][remove_c][0] = -1
                            subset[small_j][remove_c][1] = -1
                            subset[small_j][-1][0] -= 1

                # if find no partA in the subset, create a new subset
                # 濡傛灉鑲綋缁勬垚鐨勫叧鑺傜偣A,B娌℃湁琚繛鎺ュ埌鏌愪釜浜轰綋鍒欑粍鎴愭柊鐨勪汉浣
                # ------------------------------------------------------------------
                #    1.Sort each possible connection by its score.
                #    2.The connection with the highest score is indeed a final connection.
                #    3.Move to next possible connection. If no parts of this connection have
                #    been assigned to a final connection before, this is a final connection.
                #    绗笁鐐规槸璇达紝濡傛灉涓嬩竴涓彲鑳界殑杩炴帴娌℃湁涓庝箣鍓嶇殑杩炴帴鏈夊叡浜鐐圭殑璇濓紝浼氳瑙嗕负鏈缁堢殑杩炴帴锛屽姞鍏ow
                #    4.Repeat the step 3 until we are done.
                # 璇存槑瑙侊細銆https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-2-e78ab9104fc8

                elif not found and k < len(limbSeq):
                    # Fixme: 妫鏌ヤ竴涓嬫槸鍚︽纭
                    #  鍘熷鐨勬椂鍊欐槸 k<18,鍥犱负鎴戝姞浜唋imb锛屾墍浠ユ槸24,鍥犱负鐪熸鐨刲imb鏄0~16锛屾渶鍚庝袱涓17,18鏄澶栫殑涓嶆槸limb
                    #  浣嗘槸鍚庨潰鐢籰imb鐨勬椂鍊欐病鏈夋妸榧诲瓙鍜岀溂鐫涜虫湹鐨勮繛绾跨敾涓婏紝瑕佹敼杩
                    row = -1 * np.ones((20, 2))
                    row[indexA][0] = partAs[i]
                    row[indexA][1] = connection_all[k][i][2]
                    row[indexB][0] = partBs[i]
                    row[indexB][1] = connection_all[k][i][2]
                    row[-1][0] = 2
                    row[-1][1] = connection_all[k][i][-1]  # 杩欎竴浣嶇敤鏉ヨ褰曚笂杞繛鎺imb鏃剁殑闀垮害锛岀敤鏉ヤ綔涓轰笅涓杞繛鎺ョ殑鍏堥獙鐭ヨ瘑
                    row[-2][0] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    # 涓や釜绔偣鐨勭疆淇″害+limb杩炴帴鐨勭疆淇″害
                    # print('create a new subset:  ', row, '\t')
                    row = row[np.newaxis, :, :]  # 涓轰簡杩涜concatenate锛岄渶瑕佹彃鍏ヤ竴涓酱
                    subset = np.concatenate((subset, row), axis=0)

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1][0] < 4 or subset[i][-2][0] / subset[i][-1][
            0] < 0.45:  # (params['thre1'] + params['thre2']) / 2:  # todo: tune, it matters much!
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    canvas = cv2.imread(input_image)  # B,G,R order
    # canvas = oriImg
    keypoints = []

    for s in subset[..., 0]:
        keypoint_indexes = s[:18]  # 瀹氫箟鐨刱eypoint涓鍏辨湁18涓
        person_keypoint_coordinates = []
        for index in keypoint_indexes:
            if index == -1:
                # "No candidate for keypoint" # 鏍囧織涓-1鐨刾art鏄病鏈夋娴嬪埌鐨
                X, Y = 0, 0
            else:
                X, Y = candidate[index.astype(int)][:2]
            person_keypoint_coordinates.append((X, Y))
        person_keypoint_coordinates_coco = [None] * 17

        for dt_index, gt_index in dt_gt_mapping.items():
            if gt_index is None:
                continue
            person_keypoint_coordinates_coco[gt_index] = person_keypoint_coordinates[dt_index]

        keypoints.append((person_keypoint_coordinates_coco, 1 - 1.0 / s[-2]))  # s[19] is the score

    for i in range(len(keypoints)):
        print('the {}th keypoint detection result is : '.format(i), keypoints[i])

    # 鐢绘墍鏈夌殑宄板
    # for i in range(18):
    #     #     rgba = np.array(cmap(1 - i/18. - 1./36))
    #     #     rgba[0:3] *= 255
    #     for j in range(len(all_peaks[i])):  # all_peaks淇濆瓨浜嗗潗鏍囷紝score浠ュ強id
    #         # 娉ㄦ剰x,y鍧愭爣璋佸湪鍓嶈皝鍦ㄥ悗锛屽湪杩欎釜project涓湁鐐规贩涔
    #         cv2.circle(canvas, all_peaks[i][j][0:2], 3, colors[i], thickness=-1)

    # 鐢绘墍鏈夌殑楠ㄦ灦
    color_board = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    color_idx = 0
    for i in draw_list:  # 鐢诲嚭18涓猯imb銆Fixme锛氭垜璁捐浜25涓猯imb,鐢荤殑limb椤哄簭闇瑕佽皟鏁达紝鐩稿簲color鏁颁篃瑕佸鍔
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])][..., 0]
            if -1 in index:  # 鏈-1璇存槑娌℃湁瀵瑰簲鐨勫叧鑺傜偣涓庝箣鐩歌繛,鍗虫湁涓涓被鍨嬬殑part娌℃湁缂哄け锛屾棤娉曡繛鎺ユ垚limb
                continue
            # 鍦ㄤ笂涓涓猚ell涓湁銆canvas = cv2.imread(test_image) # B,G,R order
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 3), int(angle), 0,
                                       360, 1)

            cv2.circle(cur_canvas, (int(Y[0]), int(X[0])), 4, color=[0, 0, 0], thickness=2)
            cv2.circle(cur_canvas, (int(Y[1]), int(X[1])), 4, color=[0, 0, 0], thickness=2)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[color_board[color_idx]])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        color_idx += 1
    return canvas


if __name__ == '__main__':
    input_image = args.image
    output = args.output

    posenet = NetworkEval(opt, config, bn=True)

    print('Resuming from checkpoint ...... ')

    # #################################################
    # from collections import OrderedDict
    #
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint['weights'].items():
    #     # if 'out' in k or 'merge' in k:
    #     #     continue
    #     name = 'module.' + k  # add prefix 'module.'
    #     new_state_dict[name] = v
    # posenet.load_state_dict(new_state_dict)  # , strict=False
    # # #################################################

    checkpoint = torch.load(opt.ckpt_path, map_location=torch.device('npu'))  # map to cpu to save the gpu memory
    posenet.load_state_dict(checkpoint['weights'])  # 鍔犲叆浠栦汉璁粌鐨勬ā鍨嬶紝鍙兘闇瑕佸拷鐣ラ儴鍒嗗眰锛屽垯strict=False
    print('Network weights have been resumed from checkpoint...')

    if torch.npu.is_available():
        posenet.npu()

    from apex import amp

    posenet = amp.initialize(posenet,
                             opt_level=args.opt_level,
                             keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                             loss_scale=args.loss_scale)
    posenet.eval()  # set eval mode is important

    tic = time.time()
    print('start processing...')
    # load config
    params, model_params = config_reader()
    tic = time.time()
    # generate image with body parts
    with torch.no_grad():
        canvas = process(input_image, params, model_params, config.heat_layers + 2,
                         config.paf_layers)  # todo background + 2

    toc = time.time()
    print('processing time is %.5f' % (toc - tic))

    # TODO: the prediction is slow, how to fix it? Not solved yet. see:
    #  https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation/issues/5

    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)  # cv2.WINDOW_NORMAL 鑷姩閫傚悎鐨勭獥鍙ｅぇ灏
    cv2.imshow('result', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output, canvas)

    # pdf = PdfPages(output + '.pdf')
    # plt.figure()
    # plt.plot(canvas[:, :, [2, 1, 0]])
    # plt.tight_layout()
    # plt.show()
    # pdf.savefig()
    # plt.close()
    # pdf.close()

    # dummy_input = torch.randn(1, 384, 384, 3)
    # from thop import profile
    # from thop import clever_format
    # flops, params = profile(posenet, inputs=(dummy_input,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)
