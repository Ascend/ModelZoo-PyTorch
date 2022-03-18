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

import acl
from acl_net import AclModel

import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm

DTYPE = {
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'uint64': np.uint64
}

if __name__ == '__main__':
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device_id', required=True, type=int)
    parser.add_argument('--cpu_run', required=True, choices=['True', 'False'])
    parser.add_argument('--sync_infer', required=True, choices=['True', 'False'])
    parser.add_argument('--workspace', required=True, type=int)
    parser.add_argument('--input_info_file_path', required=True)
    parser.add_argument('--input_dtypes', required=True)
    parser.add_argument('--infer_res_save_path', required=True)
    parser.add_argument('--res_save_type', required=True, choices=['bin', 'npy'])
    opt = parser.parse_args()

    # 创建模型
    measurements = {}
    om_model = AclModel(device_id=opt.device_id,
                        model_path=opt.model_path,
                        sync_infer=eval(opt.sync_infer),
                        measurements=measurements,
                        key='per_infer_time_ns',
                        cpu_run=eval(opt.cpu_run))

    # 创建目录
    if os.path.exists(opt.infer_res_save_path):
        shutil.rmtree(opt.infer_res_save_path)
    os.makedirs(opt.infer_res_save_path)

    # 读取info_file
    inputs_info = {}
    with open(opt.input_info_file_path, 'rt', encoding='utf-8') as f_info:
        line = f_info.readline()
        while line:
            line = line.rstrip('\n')
            contents = line.split()
            info = {'path': contents[1], 'shape': eval(contents[2])}
            inputs_info.setdefault(contents[0], []).append(info)
            line = f_info.readline()

    # 解析输入类型
    input_dtypes = opt.input_dtypes.split(',')
    input_dtypes = list(map(lambda x: DTYPE[x], input_dtypes))

    # 读取文件推理
    total_infer_time = 0
    total_infer_time_workspace = 0
    total_infer_num = 0
    for key, values in tqdm(inputs_info.items()):
        # 构造输入
        inputs = []
        dims = []
        for idx, value in enumerate(values):
            x = np.fromfile(value['path'], dtype=input_dtypes[idx]).reshape(value['shape'])
            inputs.append(x)
            dims.extend(value['shape'])
        dims_info = {'dimCount': len(dims), 'name': '', 'dims': dims}

        # 推理得到输出
        output = om_model(inputs, dims_info)
        total_infer_num += 1

        # 保存文件
        if opt.res_save_type == 'bin':
            for idx, data in enumerate(output):
                data.tofile(os.path.join(opt.infer_res_save_path, key + '.' + str(idx) + '.bin'))
        else:
            for idx, data in enumerate(output):
                np.save(os.path.join(opt.infer_res_save_path, key + '.' + str(idx) + '.npy'), data)

        # 计算时间
        total_infer_time += measurements['per_infer_time_ns']
        if total_infer_num > opt.workspace:
            total_infer_time_workspace += measurements['per_infer_time_ns']

    # 推理时间
    print('[INFO] Infer time:')
    msg = 'total infer num: ' + str(total_infer_num) + '\n' + \
          'total pure infer time(ms): ' + str(total_infer_time / 1000 / 1000) + '\n' + \
          'average pure infer time(ms): ' + str(total_infer_time / total_infer_num / 1000 / 1000) + '\n' + \
          'average pure infer time after workspace(ms): ' + str(abs(
        total_infer_time_workspace / (total_infer_num - opt.workspace) / 1000 / 1000)) + '\n'
    print(msg)
    with open(os.path.join(opt.infer_res_save_path, 'infer_time.txt'), 'wt', encoding='utf-8') as f_infer_time:
        f_infer_time.write(msg)
