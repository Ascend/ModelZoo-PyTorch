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
More densely connected human skeleton.
But the final performance is under estimation probably due to the unreasonable redundant limbs.
"""
import numpy as np


class TrainingOpt:
    batch_size = 5  # for single process 鏁翠釜鍒嗗竷寮忔ā鍨嬫荤殑 batch size 鏄 batch_size*world_size
    learning_rate = 1e-4  # 1e-4  # 2.5e-4  for single process 鏁翠釜鍒嗗竷寮忔ā鍨嬫荤殑鏄痩earning_rate*world_size
    config_name = "Canonical"
    hdf5_train_data = "./data/dataset/coco/link2coco2017/coco_train_dataset384.h5"
    hdf5_val_data = "./data/dataset/coco/link2coco2017/coco_val_dataset384.h5"
    nstack = 3  # stacked number of hourglass
    hourglass_inp_dim = 384  # 256  # input tensor channels fed into the hourglass block
    increase = 192  # 128 #  increased channels once down-sampling in the hourglass networks
    nstack_weight = [1, 1, 1]  # weight the losses between different stacks, stack 1, stack 2, stack 3...
    scale_weight = [0.2, 0.1, 0.4, 1, 4]  # weight the losses between different scales, scale 128, scale 64, scale 32...
    multi_task_weight = 0.2  # 0.2  # person mask loss vs keypoint loss
    keypoint_task_weight = 6  # 1 keypoint heatmap loss vs body part heatmap loss
    ckpt_path = './link2checkpoints_distributed/PoseNet_46_epoch.pth'


class TransformationParams:
    """ Hyper-parameters """
    def __init__(self, stride):
        #  TODO: tune # https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation/issues/16
        #   We will firstly scale picture so that the height of the main person always will be 0.6 of picture.
        self.target_dist = 0.6
        self.scale_prob = 0.8  # scale probability, 0: never scale, 1: always scale
        self.scale_min = 0.75  # 涔嬪墠璁粌璁剧疆鐨勬槸0.8锛屼絾鍙戠幇瀵瑰皬鐩爣寰堜笉鏄庢樉
        self.scale_max = 1.25
        self.max_rotate_degree = 40.  # todo: 鐪嬬湅hourglass涓512璁剧疆鐨勫亸绉
        self.center_perterb_max = 40.  # shift augmentation
        self.flip_prob = 0.5  # flip the image to force the network distinguish the mirror symmetrical keypoints
        self.tint_prob = 0.1  # false tint鐫鑹叉搷浣滄瘮杈冭楁椂锛屽鏋滄寜鐓0.5鐨勬鐜囪繘琛岋紝鍙兘浼氫娇寰楁瘡绉掓暟鎹墿鍏呭浘鐗囧噺灏10寮,tint瀵圭綉缁滆缁冨彲鑳芥湁璐熼潰褰卞搷
        self.sigma = 9  # 7 褰撴槸512杈撳叆鏃舵槸9
        self.keypoint_gaussian_thre = 0.005  # 0.003 浣庝簬姝ゅ肩殑gt楂樻柉鍝嶅簲鐨勫尯鍩熻缃浂
        self.limb_gaussian_thre = 0.1  # 浣庝簬姝ゅ肩殑body part gt楂樻柉鍝嶅簲鐨勫尯鍩熻缃浂
        self.paf_sigma = 7  # 5 todo: sigma of PAF 瀵逛簬PAF鐨勫垎甯冿紝璁惧叾鏍囧噯宸负澶氬皯鏈鍚堥傚憿
        # the value of sigma is important, there should be an equal contribution between foreground
        # and background heatmap pixels. Otherwise, there is a prior towards the background that forces the
        # network to converge to zero.
        self.paf_thre = 1 * stride  # equals to 1.0 * stride in this program, used to include the end-points of limbs 
        #  涓轰簡鐢熸垚鍦≒AF鏃讹紝璁＄畻limb绔偣杈圭晫鏃朵娇鐢紝鍦ㄦ渶鍚庝竴涓猣eature map涓
        # 灏嗕笅鐣屽線涓嬪亸绉1*stride鍍忕礌璐紝鎶婁笂鐣屽線涓婂亸绉1*stride涓儚绱犲


class CanonicalConfig:
    """Config used in ouf project"""
    def __init__(self):
        self.width = 384
        self.height = 384
        self.stride = 4  # 鐢ㄤ簬璁＄畻缃戠粶杈撳嚭鐨刦eature map鐨勫昂瀵
        # self.img_mean = [0.485, 0.456, 0.406]  # RGB format mean and standard variance
        # self.img_std = [0.229, 0.224, 0.225]
        self.parts = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank",
                      "Lhip", "Lkne", "Lank", "Reye", "Rear", "Leye", "Lear"]  # , "navel"

        self.num_parts = len(self.parts)
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))
        # help the network to detect body parts
        self.parts += ["background"]  # person mask浣滀负鑳屾櫙涔嬩竴, global config index: 42
        # force the network to learn to distinguish the keypoints from background
        self.parts += ["reverseKeypoint"]  # 瀵规墍鏈塳eypoints鍙栧弽浣滀负鑳屾櫙浜, global config index: 43
        self.num_parts_with_background = len(self.parts)
        self.leftParts, self.rightParts = CanonicalConfig.ltr_parts(self.parts_dict)

        # this numbers probably copied from matlab they are 1.. based not 0.. based
        self.limb_from = ["neck", "neck", "neck", "neck", "neck", "nose", "Reye", "nose", "Leye", "nose", "nose",
                          "Reye", "neck", "nose", "Rear",
                          "neck", "nose", "Lear", "Rsho", "neck", "Lsho", "neck", "Relb", "Relb", "Rsho", "Lelb",
                          "Lsho", "neck", "Rsho", "Lsho", "neck", "Lsho", "Rsho", "Rhip", "Rwri", "Lwri", "Rhip",
                          "Lhip", "Rsho", "Lhip", "Rhip",
                          "Lsho", "Rkne", "Rkne", "Rhip", "Lkne", "Lkne", "Lhip", "Rkne"]

        self.limb_to = ["nose", "Reye", "Rear", "Leye", "Lear", "Reye", "Rear", "Leye", "Lear", "Rear", "Lear", "Leye",
                        "Rsho", "Rsho", "Rsho",
                        "Lsho", "Lsho", "Lsho", "Relb", "Relb", "Lelb", "Lelb", "Lelb", "Rwri", "Rwri", "Lwri", "Lwri",
                        "Rhip", "Rhip", "Rhip",
                        "Lhip", "Lhip", "Lhip", "Lhip", "Rhip", "Lhip", "Rkne", "Rkne", "Rkne", "Lkne", "Lkne", "Lkne",
                        "Lkne", "Rank", "Rank",
                        "Rank", "Lank", "Lank", "Lank"]

        self.limb_from = [self.parts_dict[n] for n in self.limb_from]
        self.limb_to = [self.parts_dict[n] for n in self.limb_to]

        assert self.limb_from == [x for x in
                                  [1, 1, 1, 1, 1, 0, 14, 0, 16, 0, 0, 14, 1, 0, 15, 1, 0, 17, 2, 1, 5, 1, 3, 3, 2, 6, 5,
                                   1, 2, 5, 1, 5, 2, 8, 4, 7, 8, 11, 2, 11, 8, 5, 9, 9, 8, 12, 12, 11, 9]]
        assert self.limb_to == [x for x in
                                [0, 14, 15, 16, 17, 14, 15, 16, 17, 15, 17, 16, 2, 2, 2, 5, 5, 5, 3, 3, 6, 6, 6, 4, 4,
                                 7, 7, 8, 8, 8, 11, 11, 11, 11, 8, 11, 9, 9, 9, 12, 12, 12, 12, 10, 10, 10, 13, 13, 13]]

        self.limbs_conn = list(zip(self.limb_from, self.limb_to))

        self.paf_layers = len(self.limbs_conn)
        self.heat_layers = self.num_parts
        # layers of keypoint and body part heatmaps PLUS ++ 2 background
        self.num_layers = self.paf_layers + self.heat_layers + 2

        self.paf_start = 0
        self.heat_start = self.paf_layers  # Notice: 姝ゅchannel瀹夋帓涓婏紝paf_map鍦ㄥ墠锛宧eat_map鍦ㄥ悗
        self.bkg_start = self.paf_layers + self.heat_layers  # 鐢ㄤ簬feature map鐨勮鏁,2涓猙ackground鐨勮捣濮嬬偣

        self.offset_layers = 2 * self.num_parts
        self.offset_start = self.num_layers

        self.mask_shape = (self.height // self.stride, self.width // self.stride)  # 46, 46
        self.parts_shape = (self.height // self.stride, self.width // self.stride, self.num_layers)  # 46, 46, 59
        self.offset_shape = (self.height // self.stride, self.width // self.stride, self.offset_layers)

        self.transform_params = TransformationParams(self.stride)

        # ####################### Some configurations only used in inference process  ###########################
        # map between original coco keypoint ids and  our keypoint ids
        # 鍥犱负CMU鐨勫畾涔夊拰COCO瀹樻柟瀵筳oint缂栧彿鐨勫畾涔変笉鐩稿悓锛屾墍浠ラ渶瑕侀氳繃mapping鎶婄紪鍙锋敼杩囨潵銆銆
        self.dt_gt_mapping = {0: 0, 1: None, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12, 9: 14, 10: 16, 11: 11, 12: 13,
                              13: 15, 14: 2, 15: 1, 16: 4, 17: 3}  # , 18: None 娌℃湁浣跨敤鑲氳剱

        # For the flip augmentation in the inference process only
        self.flip_heat_ord = np.array([0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 16, 17, 14, 15, 18, 19])
        self.flip_paf_ord = np.array(
            [0, 3, 4, 1, 2, 7, 8, 5, 6, 10, 9, 11, 15, 16, 17, 12, 13, 14, 20, 21, 18, 19, 22, 25, 26, 23, 24, 30, 31,
             32, 27, 28, 29, 33, 35, 34, 39, 40, 41, 36, 37, 38, 42, 46, 47, 48, 43, 44, 45])
        self.draw_list = [0, 5, 7, 6, 8, 12, 18, 23, 15, 20, 25, 27, 36, 43, 30, 39, 46, 33]  # to draw skeleton
        # #########################################################################################################

    @staticmethod
    # staticmethod淇グ鐨勬柟娉曞畾涔変笌鏅氬嚱鏁版槸涓鏍风殑, staticmethod鏀寔绫诲璞℃垨鑰呭疄渚嬪鏂规硶鐨勮皟鐢,鍗冲彲浣跨敤A.f()鎴栬卆.f()
    def ltr_parts(parts_dict):
        # When we flip image left parts became right parts and vice versa.
        # This is the list of parts to exchange each other.
        leftParts = [parts_dict[p] for p in ["Lsho", "Lelb", "Lwri", "Lhip", "Lkne", "Lank", "Leye", "Lear"]]
        rightParts = [parts_dict[p] for p in ["Rsho", "Relb", "Rwri", "Rhip", "Rkne", "Rank", "Reye", "Rear"]]
        return leftParts, rightParts


class COCOSourceConfig:
    """Original config used in COCO dataset"""
    def __init__(self, hdf5_source):
        """
        Instantiate a COCOSource Config object锛
        :param hdf5_source: the path only of hdf5 training materials generated by coco_mask_hdf5.py
        """
        self.hdf5_source = hdf5_source
        self.parts = ['nose', 'Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb',
                      'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank',
                      'Rank']  # coco鏁版嵁闆嗕腑鍏抽敭鐐圭被鍨嬪畾涔夌殑椤哄簭

        self.num_parts = len(self.parts)

        # for COCO neck is calculated like mean of 2 shoulders.
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))

    def convert(self, meta, global_config):
        """Convert COCO configuration (joint annotation) into ours configuration of this project"""
        # ----------------------------------------------
        # ---灏哻oco config涓鏁版嵁鐨勫畾涔夋敼鎴怌MU椤圭洰涓殑鏍煎紡---
        # ----------------------------------------------

        joints = np.array(meta['joints'])

        assert joints.shape[1] == len(self.parts)

        result = np.zeros((joints.shape[0], global_config.num_parts, 3))
        # result鏄竴涓笁缁存暟缁勶紝shape[0]鍜屼汉鏁版湁鍏筹紝姣忎竴琛屽嵆shape[1]鍜屽叧鑺傜偣鏁扮洰鏈夊叧锛屾渶鍚庝竴缁村害闀垮害涓3,鍒嗗埆鏄痻,y,v,鍗冲潗鏍囧煎拰鍙鏍囧織浣
        result[:, :, 2] = 3.
        # OURS - # 3 never marked up in this dataset, 2 - not marked up in this person, 1 - marked and visible,
        # 0 - marked but invisible. 鍏充簬visible鍊肩殑閲嶆柊瀹氫箟鍦╟oco_mask_hdf5.py涓畬鎴愪簡

        for p in self.parts:
            coco_id = self.parts_dict[p]

            if p in global_config.parts_dict:
                global_id = global_config.parts_dict[p]  # global_id鏄湪璇ラ」鐩腑浣跨敤鐨勫叧鑺傜偣缂栧彿锛屽洜涓洪澶栧姞鍏ヤ簡neck(navel?)锛屼笌鍘熷coco鏁版嵁闆嗕腑瀹氫箟涓嶅悓
                assert global_id != 1, "neck shouldn't be known yet"
                # assert global_id != 2, "navel shouldn't be known yet"
                result[:, global_id, :] = joints[:, coco_id, :]

        if 'neck' in global_config.parts_dict:  # neck point works as a root note
            neckG = global_config.parts_dict['neck']
            # parts_dict['neck']銆锛濄锛, parts_dict鏄墠闈㈠畾涔夎繃鐨勫瓧鍏哥被鍨嬶紝鑺傜偣鍚嶇О锛氬簭鍙
            RshoC = self.parts_dict['Rsho']
            LshoC = self.parts_dict['Lsho']

            # no neck in coco database, we calculate it as average of shoulders
            #  here, we use 0 - hidden, 1 visible, 2 absent to represent the visibility of keypoints
            #  - it is not the same as coco values they processed by generate_hdf5

            # -------------------------------鍘熷coco鍏充簬visible鏍囩鐨勫畾涔夛紞锛嶏紞锛嶏紞锛嶏紞锛嶏紞--------------锛嶏紞锛嶏紞锛嶏紞锛嶏紞锛#
            # 绗笁涓厓绱犳槸涓爣蹇椾綅v锛寁涓0鏃惰〃绀鸿繖涓叧閿偣娌℃湁鏍囨敞锛堣繖绉嶆儏鍐典笅x = y = v = 0锛夛紝
            # v涓1鏃惰〃绀鸿繖涓叧閿偣鏍囨敞浜嗕絾鏄笉鍙锛堣閬尅浜嗭級锛寁涓2鏃惰〃绀鸿繖涓叧閿偣鏍囨敞浜嗗悓鏃朵篃鍙銆
            # ------------------------------------ ----------------------------銆锛嶏紞锛嶏紞锛嶏紞锛嶏紞锛嶏紞锛嶏紞锛嶏紞锛嶏紞锛嶏紞锛嶏紞锛#

            both_shoulders_known = (joints[:, LshoC, 2] < 2) & (joints[:, RshoC, 2] < 2)  # 鎸変綅杩愮畻
            # 鐢═rue鍜孎alse浣滀负绱㈠紩
            result[~both_shoulders_known, neckG, 2] = 2.  # otherwise they will be 3. aka 'never marked in this dataset'
            # ~both_shoulders_known bool绫诲瀷鎸変綅鍙栧弽
            result[both_shoulders_known, neckG, 0:2] = (joints[both_shoulders_known, RshoC, 0:2] +
                                                        joints[both_shoulders_known, LshoC, 0:2]) / 2
            result[both_shoulders_known, neckG, 2] = np.minimum(joints[both_shoulders_known, RshoC, 2],
                                                                joints[both_shoulders_known, LshoC, 2])
            # 鏈鍚庝竴浣嶆槸 visible銆鏍囧織浣嶏紝濡傛灉涓や釜鑺傜偣涓湁浠讳綍涓涓妭鐐逛笉鍙锛屽垯涓棿鑺傜偣neck璁句负涓嶅彲瑙

        if 'navel' in global_config.parts_dict:  # add navel keypoint or not?
            navelG = global_config.parts_dict['navel']
            # parts_dict['navel']銆锛 2, parts_dict鏄墠闈㈠畾涔夎繃鐨勫瓧鍏哥被鍨嬶紝鑺傜偣鍚嶇О锛氬簭鍙
            RhipC = self.parts_dict['Rhip']
            LhipC = self.parts_dict['Lhip']

            # no navel in coco database, we calculate it as average of hipulders
            both_hipulders_known = (joints[:, LhipC, 2] < 2) & (joints[:, RhipC, 2] < 2)  # 鎸変綅杩愮畻
            # 鐢═rue鍜孎alse浣滀负绱㈠紩
            result[
                ~both_hipulders_known, navelG, 2] = 2.  # otherwise they will be 3. aka 'never marked in this dataset'
            # ~both_hipulders_known bool绫诲瀷鎸変綅鍙栧弽
            result[both_hipulders_known, navelG, 0:2] = (joints[both_hipulders_known, RhipC, 0:2] +
                                                         joints[both_hipulders_known, LhipC, 0:2]) / 2
            result[both_hipulders_known, navelG, 2] = np.minimum(joints[both_hipulders_known, RhipC, 2],
                                                                 joints[both_hipulders_known, LhipC, 2])

        meta['joints'] = result

        return meta

    def repeat_mask(self, mask, global_config, joints=None):
        # 澶嶅埗mask鍒颁釜鏁板埌global_config閫氶亾鏁帮紝浣嗘槸鎴戜滑涓嶈繘琛岄氶亾鐨勫鍒讹紝鍒╃敤broadcast锛岃妭鐪佸唴瀛
        mask = np.repeat(mask[:, :, np.newaxis], global_config.num_layers, axis=2)  # mask澶嶅埗鎴愪簡57涓氶亾
        return mask

    def source(self):
        # return the path
        return self.hdf5_source


# more information on keypoints mapping is here
# https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/7

Configs = {}
Configs["Canonical"] = CanonicalConfig


def GetConfig(config_name):
    config = Configs[config_name]()  # () will instantiate an object of Configs[config_name] class

    dct = config.parts[:]
    dct = [None] * (config.num_layers - len(dct)) + dct

    for (i, (fr, to)) in enumerate(config.limbs_conn):
        name = "%s->%s" % (config.parts[fr], config.parts[to])
        print(i, name)
        x = i

        assert dct[x] is None
        dct[x] = name

    from pprint import pprint
    pprint(dict(zip(range(len(dct)), dct)))

    return config


if __name__ == "__main__":
    # test it
    foo = GetConfig("Canonical")
    print('the number of paf_layers is: %d, and the number of heat_layer is: %d' % (foo.paf_layers, foo.heat_layers))
