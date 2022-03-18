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
from configobj import ConfigObj
import numpy as np


def config_reader():
    config = ConfigObj('/home/jia/Desktop/Improved-Body-Parts/utils/config')

    param = config['param']  # 缁ф壙浜哾ict鐨勪竴绉嶅瓧鍏哥被鍨
    model_id = param['modelID']
    model = config['models'][model_id]  # 鍥犱负config鏂囦欢涓紝model閮ㄥ垎鍙堟湁涓涓猍[1]]鍒嗘敮锛屾墍浠ュ張鍔犱笂浜唌odel_id=1鐨勭储寮
    model['boxsize'] = int(model['boxsize'])
    model['stride'] = int(model['stride'])
    model['max_downsample'] = int(model['max_downsample'])
    model['padValue'] = int(model['padValue'])
    #param['starting_range'] = float(param['starting_range'])
    #param['ending_range'] = float(param['ending_range'])
    param['remove_recon'] = int(param['remove_recon'])
    param['use_gpu'] = int(param['use_gpu'])
    param['starting_range'] = float(param['starting_range'])
    param['ending_range'] = float(param['ending_range'])
    param['scale_search'] = list(map(float, param['scale_search']))  # [float(param['scale_search'])]  #     #
    param['rotation_search'] = list(map(float, param['rotation_search']))  # [float(param['scale_search'])]  #     #
    param['thre1'] = float(param['thre1'])
    param['thre2'] = float(param['thre2'])

    param['connect_ration'] = float(param['connect_ration'])
    param['connection_tole'] = float(param['connection_tole'])
    param['mid_num'] = int(param['mid_num'])
    param['min_num'] = int(param['min_num'])
    param['len_rate'] = float(param['len_rate'])
    param['offset_radius'] = int(param['offset_radius'])
    param['crop_ratio'] = float(param['crop_ratio'])
    param['bbox_ratio'] = float(param['bbox_ratio'])
    param['GPUdeviceNumber'] = int(param['GPUdeviceNumber'])

    return param, model


if __name__ == "__main__":
    config_reader()
